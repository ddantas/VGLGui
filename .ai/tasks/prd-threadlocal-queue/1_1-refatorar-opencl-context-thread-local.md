---
id: 1.1
fase: 1
titulo: Refatorar vgl_lib/opencl_context.py com queue thread-local
requisitos: [RF-001, RF-002, RF-003]
depende: —
paralela: false
---

# 1.1 — Refatorar `vgl_lib/opencl_context.py` com queue thread-local

## Contexto

`VglClContext.commandQueue` é hoje um atributo copiado na inicialização (snapshot da
main thread). Para suportar execução paralela futura, cada thread precisa de sua própria
`cl.CommandQueue`. A mudança deve ser 100% transparente — nenhum caller é alterado.

Ver TechSpec §3 para o design completo e ADR-001 para justificativa da abordagem.

## Arquivos

- **Modificar:** `vgl_lib/opencl_context.py`
- **NÃO tocar:** qualquer outro arquivo do projeto

## O que implementar

### 1. Classe `VglClContext` — substituir atributo por property

**Antes:**
```python
class VglClContext:
    def __init__(self, pl, dv, cn, cq):
        self.platformId   = pl
        self.deviceId     = dv
        self.context      = cn
        self.commandQueue = cq     # cópia estática
```

**Depois:**
```python
class VglClContext:
    def __init__(self, pl, dv, cn, ocl_ctx):
        self.platformId = pl
        self.deviceId   = dv
        self.context    = cn
        self._ocl_ctx   = ocl_ctx  # referência viva para opencl_context

    @property
    def commandQueue(self):
        return self._ocl_ctx.queue

    @property
    def queue(self):
        return self.commandQueue
```

### 2. Classe `opencl_context` — adicionar threading.local()

Em `__init__`, substituir `self.queue = cl.CommandQueue(self.ctx)` por:
```python
import threading
self._local = threading.local()
```

Adicionar as properties e método:
```python
@property
def queue(self):
    if not hasattr(self._local, 'queue'):
        import threading as _th
        print(f"[opencl] nova CommandQueue para thread '{_th.current_thread().name}'")
        self._local.queue = cl.CommandQueue(self.ctx)
    return self._local.queue

@property
def commandQueue(self):
    return self.queue

def finish_queue(self):
    if hasattr(self._local, 'queue'):
        self._local.queue.finish()
```

### 3. Atualizar `get_vglClContext_attributes()`

```python
def get_vglClContext_attributes(self):
    return VglClContext(
        self.platform.int_ptr,
        self.device.int_ptr,
        self.ctx,
        self          # passa self, não self.queue
    )
```

## Padrão de referência

- `vgl_lib/opencl_context.py` — arquivo a modificar (ler antes de editar)
- `vgl_lib/vglClUtil.py` linhas 41, 62, 70 — exemplos de callers que NÃO devem mudar

## Acceptance Criteria

- AC-A: `id(vl.get_ocl().commandQueue)` retorna valores diferentes em 2 threads distintas
- AC-B: `id(vl.get_ocl().commandQueue)` retorna o mesmo valor em 2 chamadas da mesma thread
- AC-C: `vl.get_ocl_context().finish_queue()` executa sem exceção na main thread
- AC-D: `git diff --name-only` mostra apenas `vgl_lib/opencl_context.py`

## Verificar

```bash
cd /home/joao/Documents/TCC_1/VGLGui
git diff --name-only   # deve mostrar só vgl_lib/opencl_context.py
```
