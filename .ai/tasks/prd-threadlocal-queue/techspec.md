---
id: techspec-threadlocal-queue
prd: prd-threadlocal-queue
status: aprovado
created: 2026-03-16
complexity: Small
---

# TechSpec — CommandQueue Thread-Local no Contexto OpenCL

## Resumo Executivo

**Feature:** CommandQueue thread-local em `vgl_lib/opencl_context.py`
**Complexidade:** Small
**Estimativa:** 1 dia de implementação + validação

**Decisões Principais:**
1. `VglClContext.commandQueue` vira `@property` delegando para `opencl_context.queue` — zero mudanças nos callers
2. `opencl_context.queue` usa `threading.local()` — cria uma `CommandQueue` por thread automaticamente

**Componentes:** 0 criados, 1 modificado (`opencl_context.py`)
**Critérios de verificação:** 7 definidos
**Riscos críticos:** 1 (GPU sem suporte a múltiplas filas — baixa probabilidade)

**Dependências bloqueantes:** nenhuma
**Questões em aberto:** 1 (não bloqueante)

---

## 1. Background e Motivação

### Por que CommandQueue não pode ser compartilhada?

A GPU não executa operações na hora que você pede — ela tem uma **fila de pedidos**
(`CommandQueue`). Você empurra tarefas nessa fila, e a GPU executa na ordem.

A especificação OpenCL define `cl_command_queue` como **não thread-safe**: se duas
threads enviam comandos para a mesma fila simultaneamente, o resultado é
comportamento indefinido (crash ou dados corrompidos).

```
Thread 1: "GPU, aplica dilate imagem A" → entra na fila...
Thread 2: interrompe e escreve "erode imagem B" no meio...
Resultado: "dilate imag[corrompido]ode imagem B" → crash
```

O `cl.Context` (a "cozinha") é thread-safe e pode ser compartilhado.
Só a fila (`CommandQueue`) precisa ser isolada por thread.

### Situação atual

`vgl_lib/opencl_context.py` mantém uma única `cl.CommandQueue` global:

```python
# opencl_context.__init__
self.queue = cl.CommandQueue(self.ctx)   # uma fila para todo o processo
```

`VglClContext` (retornado por `vl.get_ocl()`) armazena uma cópia estática:

```python
class VglClContext:
    def __init__(self, pl, dv, cn, cq):
        self.commandQueue = cq           # snapshot — aponta para fila da main thread
```

Todos os callers acessam via `vl.get_ocl().commandQueue` — ex.:
```python
# vglClUtil.py linha 41
cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, _kernel, ...)
```

Com execução serial (estado atual), não há problema. Para execução paralela
de ramos independentes (próximo PRD), múltiplas threads chamariam essas funções
simultaneamente → crash garantido.

### Esta TechSpec é pré-requisito de

`prd-paralelismo` — execução paralela de ramos independentes do DAG.

---

## 2. Escopo

### Em escopo
- Modificar `VglClContext` e `opencl_context` em `vgl_lib/opencl_context.py`
- Adicionar `finish_queue()` para sincronização explícita
- Nenhum outro arquivo modificado

### Fora de escopo
- Implementação do executor paralelo (`prd-paralelismo`)
- Suporte a múltiplos dispositivos GPU
- Modificação de `vglClUtil.py`, `vglClImage.py` ou qualquer caller

---

## 3. Design

### Componente: `opencl_context` (modificado)

**Localização:** `vgl_lib/opencl_context.py`

```python
import threading

class opencl_context:
    def __init__(self, device_type):
        # ... código existente de seleção de plataforma/device ...
        self.ctx = cl.Context([self.device])
        self._local = threading.local()     # ← NOVO: estado por thread
        self.programs = []

    @property
    def queue(self):
        """Retorna a CommandQueue da thread atual, criando se necessário."""
        if not hasattr(self._local, 'queue'):
            import threading as _th
            print(f"[opencl] nova CommandQueue para thread '{_th.current_thread().name}'")
            self._local.queue = cl.CommandQueue(self.ctx)
        return self._local.queue

    @property
    def commandQueue(self):
        """Alias para retrocompatibilidade com VglClContext legado."""
        return self.queue

    def finish_queue(self):
        """Aguarda conclusão de todas as operações da fila da thread atual."""
        if hasattr(self._local, 'queue'):
            self._local.queue.finish()

    def get_vglClContext_attributes(self):
        """Retorna VglClContext com referência viva ao opencl_context."""
        return VglClContext(
            self.platform.int_ptr,
            self.device.int_ptr,
            self.ctx,
            self              # ← passa self, não self.queue
        )

    # ... demais métodos sem alteração ...
```

### Componente: `VglClContext` (modificado)

```python
class VglClContext:
    def __init__(self, pl, dv, cn, ocl_ctx):
        self.platformId = pl
        self.deviceId   = dv
        self.context    = cn
        self._ocl_ctx   = ocl_ctx   # referência viva para opencl_context

    @property
    def commandQueue(self):
        """Delega para a fila da thread atual."""
        return self._ocl_ctx.queue

    @property
    def queue(self):
        """Alias de commandQueue para uniformidade."""
        return self.commandQueue
```

### Fluxo por thread

```
Main thread (inicialização)
  └─ vglClInit() → opencl_context() → _local = threading.local()
  └─ get_vglClContext_attributes() → VglClContext(..., self)
  └─ set_ocl(vgl_cl_ctx)  ← global ocl configurado

Main thread (execução serial — sem mudança de comportamento)
  └─ vl.get_ocl().commandQueue
      └─ VglClContext.commandQueue (property)
          └─ opencl_context.queue (property)
              └─ _local.queue existe? Sim → retorna fila da main thread

Thread paralela (futuro — prd-paralelismo)
  └─ vl.get_ocl().commandQueue
      └─ VglClContext.commandQueue (property)
          └─ opencl_context.queue (property)
              └─ _local.queue existe? Não → cria cl.CommandQueue(self.ctx)
              └─ retorna nova fila isolada para esta thread
  └─ ... executa kernel ...
  └─ vl.get_ocl_context().finish_queue()  ← aguarda GPU terminar
```

---

## 4. ADRs

Ver `adr/001-property-dinamica-vs-lock.md`.

**Resumo da decisão:** `@property` dinâmica no `VglClContext` delegando para
`opencl_context` com `threading.local()`. Alternativas rejeitadas: lock global
(impede paralelismo) e VglClContext por thread (exige mudar callers).

---

## 5. Riscos

**Risco:** GPU ou driver não suporta múltiplas `CommandQueue` no mesmo `cl.Context`.
**Probabilidade:** Baixa (suporte padrão desde OpenCL 1.0).
**Impacto:** Médio — execução paralela falha na criação da segunda fila.
**Mitigação:** Capturar `pyopencl.RuntimeError` na criação da fila; logar e relançar
com mensagem clara. Testar no hardware disponível antes de ativar paralelismo.

---

## 6. Critérios de Verificação

| Requisito | Critério | Tipo |
|---|---|---|
| RF-001 | Dois `id(vl.get_ocl().commandQueue)` em threads distintas são diferentes | Funcional |
| RF-001 | Mesmo `id()` na mesma thread em duas chamadas consecutivas | Funcional |
| RF-002 | Output binário de `demo.wksp` idêntico antes/depois da mudança | Regressão |
| RF-002 | `git diff --name-only` mostra apenas `vgl_lib/opencl_context.py` | Estrutural |
| RF-003 | `vl.get_ocl_context().finish_queue()` na main thread sem exceção | Funcional |
| RF-003 | `finish_queue()` em thread que nunca usou queue — sem exceção | Funcional |
| RNF-001 | Log de criação da fila aparece uma única vez por thread | Observabilidade |

### Script de verificação

```python
# test_threadlocal_queue.py — rodar após vglClInit()
import threading
import vgl_lib as vl

results = {}

def capture(tid):
    q1 = id(vl.get_ocl().commandQueue)
    q2 = id(vl.get_ocl().commandQueue)
    results[tid] = (q1, q2)

t1 = threading.Thread(target=capture, args=("t1",))
t2 = threading.Thread(target=capture, args=("t2",))
t1.start(); t2.start()
t1.join(); t2.join()

assert results["t1"][0] == results["t1"][1], "FALHOU: fila diferente na mesma thread"
assert results["t2"][0] == results["t2"][1], "FALHOU: fila diferente na mesma thread"
assert results["t1"][0] != results["t2"][0], "FALHOU: mesma fila em threads diferentes"
vl.get_ocl_context().finish_queue()   # não deve lançar exceção
print("OK: todas as verificações passaram")
```

---

## 7. Rollout

| Aspecto | Estratégia |
|---|---|
| Rollback | `git revert` — 1 arquivo, sem schema/migration |
| Validação | Rodar `demo.wksp` e `fundus.wksp`; diff binário das saídas vs. baseline |
| Feature flag | Não necessário — mudança transparente para execução serial |

---

## 8. Questões em Aberto

| # | Questão | Impacto | Bloqueante? |
|---|---|---|---|
| 1 | GPU do ambiente de desenvolvimento suporta múltiplas CommandQueues no mesmo Context? | Médio | Não — execução serial continua funcionando independente da resposta |

---

## Self-Audit

| Requisito | Coberto? | Seção | AC testável? | Critério de verificação? |
|---|---|---|---|---|
| RF-001 | ✅ | §3 | ✅ | ✅ §6 |
| RF-002 | ✅ | §3 | ✅ | ✅ §6 |
| RF-003 | ✅ | §3 | ✅ | ✅ §6 |
| RNF-001 | ✅ | §6 | ✅ (log) | ✅ §6 |
| RNF-002 | ✅ | §6 | ✅ (diff) | ✅ §6 |
| RNF-003 | ⚠️ | §5 (risco) | Parcial | Não explícito |

**RNF-003 (sem vazamento):** Python gerencia ciclo de vida via GC. Filas de threads encerradas
serão coletadas quando a thread terminar e `_local` for liberado. Sem ação adicional necessária.
