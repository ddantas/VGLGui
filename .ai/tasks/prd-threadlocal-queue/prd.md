---
id: prd-threadlocal-queue
title: CommandQueue Thread-Local no Contexto OpenCL (VGL_CL)
status: draft
created: 2026-03-16
classification: Small
depends-on: —
blocks: prd-paralelismo
---

# PRD — CommandQueue Thread-Local no Contexto OpenCL

## Contexto: O que é Thread-Safe? (e por que importa)

### O que é uma thread?

Imagine que seu programa é uma cozinha. Normalmente tem **um cozinheiro** fazendo
tudo em sequência: corta o legume, depois cozinha, depois tempera.

**Threads** são como ter **múltiplos cozinheiros** na mesma cozinha ao mesmo tempo —
cada um executando uma parte do trabalho em paralelo.

### O que é a "fila" (CommandQueue)?

A GPU não executa operações na hora que você pede. Ela tem uma **fila de pedidos** —
você empurra tarefas nessa fila, e a GPU vai executando uma por vez na ordem que
chegaram. É como o balcão de uma lanchonete: você faz o pedido, ele vai para a fila,
e a cozinha executa na ordem.

```
Thread 1: "GPU, aplica dilate na imagem A"  → entra na fila
Thread 2: "GPU, aplica erode na imagem B"   → entra na fila
GPU executa: dilate... erode...
```

### O problema: a fila não é thread-safe

"Thread-safe" significa que um recurso **pode ser usado por múltiplas threads ao
mesmo tempo sem dar problema**.

A `CommandQueue` do OpenCL **não é thread-safe** porque internamente ela usa uma
estrutura de dados simples sem proteção. Se duas threads tentam empurrar pedidos
na fila **ao mesmo tempo**, pode acontecer isso:

```
Thread 1 está escrevendo "dilate imagem A" na fila...
Thread 2 interrompe e escreve "erode imagem B" no meio...
Resultado na fila: "dilate imag[corrompido]ode imagem B"
           → crash ou resultado errado
```

É como dois garçons tentando escrever no mesmo papel de pedido ao mesmo tempo.

### A solução: uma fila por cozinheiro

Em vez de todos os cozinheiros compartilharem **o mesmo balcão**, cada um tem
**o seu próprio balcão** — mas todos mandam os pedidos para **a mesma cozinha** (GPU).

```
Thread 1 → fila própria → ┐
Thread 2 → fila própria → ├→ GPU (contexto compartilhado)
Thread 3 → fila própria → ┘
```

O `cl.Context` (a "cozinha") é thread-safe — pode ser compartilhado. Só a fila
que não pode. Com `threading.local()`, cada thread cria automaticamente **sua
própria fila** na primeira vez que precisa, sem nenhuma mudança no resto do código.

### Resumo visual

| | Situação atual | Com a mudança |
|---|---|---|
| Threads | 1 (sequencial) | N (paralelas) |
| Filas | 1 compartilhada | 1 por thread |
| Contexto GPU | 1 compartilhado | 1 compartilhado |
| Thread-safe? | Não se paralelo | ✅ Sim |

---

## 1. Resumo Executivo

A classe `opencl_context` em `vgl_lib/opencl_context.py` mantém uma única
`cl.CommandQueue` compartilhada globalmente. A especificação OpenCL define
`cl_command_queue` como não thread-safe: submeter comandos de múltiplas threads
para a mesma fila é comportamento indefinido (crash ou resultado corrompido).
Este PRD especifica a substituição da fila única por filas thread-local —
uma fila criada automaticamente por thread — mantendo o `cl.Context` compartilhado
(que é thread-safe). Esta mudança é **pré-requisito** para o PRD de paralelismo
de ramos independentes (`prd-paralelismo`).

---

## 2. Contexto e Motivação

- `vgl_lib/opencl_context.py` linha 39: `self.queue = cl.CommandQueue(self.ctx)` —
  uma única fila para todo o processo.
- Todas as funções VGL_CL acessam essa fila via `vl.get_ocl().commandQueue`
  (vglClUtil.py linhas 41, 62, 70, 76; vglClImage.py linhas 135, 156, 191, 201, 366, 376).
- Com execução serial (estado atual), não há problema — só uma thread usa a fila.
- Para executar ramos independentes em paralelo (próximo PRD), múltiplas threads
  precisarão chamar funções VGL_CL simultaneamente → crash garantido com a fila atual.
- A solução `threading.local()` cria a fila na primeira chamada de cada thread,
  reutilizando-a nas chamadas seguintes da mesma thread. O `cl.Context` permanece
  único e compartilhado (conforme a spec OpenCL, contextos são thread-safe).

---

## 3. Escopo e Non-Goals

### Em escopo

- Modificar `opencl_context.py` para usar `threading.local()` na propriedade `queue`
- Manter retrocompatibilidade total: `vl.get_ocl().commandQueue` continua funcionando
- Garantir que a thread principal (execução serial atual) não tenha regressão
- Adicionar método `finish_queue()` para sincronização explícita pós-operação paralela

### Non-Goals

| ID | Non-Goal | Motivo |
|----|----------|--------|
| NG-001 | Implementar execução paralela de nós | Escopo do prd-paralelismo |
| NG-002 | Modificar vglClUtil.py ou vglClImage.py | A mudança deve ser transparente para o restante |
| NG-003 | Suporte a múltiplos dispositivos GPU | Um dispositivo por contexto (comportamento atual mantido) |
| NG-004 | Out-of-order CommandQueue | Não resolve thread-safety; complexidade desnecessária |
| NG-005 | Alterar formato `.wksp` ou parser | Não relacionado |

---

## 4. Requisitos Funcionais

---

### RF-001 [MUST]: Fila Thread-Local

O sistema DEVE criar uma `cl.CommandQueue` independente para cada thread que
acessar `vl.get_ocl().commandQueue` ou `vl.get_ocl().queue`.

**Acceptance Criteria:**
- AC-001-A: Duas threads distintas que acessam `vl.get_ocl().queue` recebem
  objetos `cl.CommandQueue` diferentes (verificável por `id()`).
- AC-001-B: A mesma thread que acessa `vl.get_ocl().queue` duas vezes recebe
  o mesmo objeto (não cria uma nova fila a cada chamada).
- AC-001-C: A thread principal (execução serial do workflow) continua recebendo
  uma fila válida sem nenhuma alteração no código chamador.

---

### RF-002 [MUST]: Retrocompatibilidade Total

O atributo `commandQueue` DEVE continuar funcionando como antes em todo código
que já o utiliza, sem nenhuma alteração nesses arquivos.

**Acceptance Criteria:**
- AC-002-A: `vl.get_ocl().commandQueue` retorna uma `cl.CommandQueue` válida
  em qualquer thread (mesmo comportamento de antes para código serial).
- AC-002-B: Nenhum arquivo fora de `opencl_context.py` precisa ser modificado
  para que a mudança funcione.
- AC-002-C: Workflows existentes (demo.wksp, fundus.wksp, etc.) executam com
  resultado idêntico ao atual após a mudança.

---

### RF-003 [MUST]: Sincronização Explícita

O sistema DEVE fornecer um método para aguardar a conclusão de todas as operações
enfileiradas na fila da thread atual.

**Acceptance Criteria:**
- AC-003-A: Existe um método `finish_queue()` (ou equivalente) em `opencl_context`
  que chama `queue.finish()` na fila da thread corrente.
- AC-003-B: Chamar `finish_queue()` na thread principal não lança exceção.
- AC-003-C: Chamar `finish_queue()` em uma thread que ainda não criou fila
  não lança exceção (cria e finaliza imediatamente).

---

## 5. Requisitos Não-Funcionais

| ID | Requisito | Threshold |
|----|-----------|-----------|
| RNF-001 | Overhead de criação da fila | < 50ms por thread (criação única, não por operação) |
| RNF-002 | Resultado dos workflows | Binariamente idêntico ao atual para execução serial |
| RNF-003 | Sem vazamento de recursos | Filas de threads encerradas não devem manter referências vivas indefinidamente |

---

## 6. Critérios de Sucesso

| Métrica | Baseline | Target | Método |
|---------|----------|--------|--------|
| Workflows existentes produzem resultado idêntico | — | 100% idêntico | diff binário das imagens de saída |
| Acesso thread-safe confirmado | Crash em execução paralela | Sem crash em 10 execuções paralelas de teste | script de teste com 2 threads |
| Nenhum arquivo alterado além de opencl_context.py | — | 0 outros arquivos modificados | git diff |

---

## 7. Riscos e Dependências

| ID | Risco | Probabilidade | Impacto | Mitigação |
|----|-------|---------------|---------|-----------|
| R-001 | GPUs com drivers antigos podem não suportar múltiplas filas no mesmo contexto | Baixa | Médio | Testar no hardware disponível; fallback para fila única com lock se necessário |
| R-002 | Threads Python do DPG (render loop) podem criar filas fantasmas | Baixa | Baixo | Filas não utilizadas têm overhead desprezível |

**Dependências:**
- `vgl_lib/opencl_context.py` — único arquivo a modificar
- `vgl_lib/__init__.py` — verificar como `get_ocl()` expõe o contexto global

---

## 8. Proteções — NÃO ALTERAR

- **`vglClUtil.py`**: nenhuma modificação — deve continuar funcionando sem mudanças
- **`vglClImage.py`**: nenhuma modificação
- **Todos os arquivos fora de `vgl_lib/opencl_context.py`**: zero mudanças
- **Interface pública**: `vl.get_ocl().commandQueue` e `vl.get_ocl().queue` mantêm a mesma assinatura
