# ADR-001: Property dinâmica vs. alternativas para CommandQueue thread-safe

**Status:** Proposto
**Data:** 2026-03-16

## Contexto

`VglClContext.commandQueue` é um atributo copiado na inicialização — um snapshot da fila
da main thread. Para suportar execução paralela de nós VGL_CL, cada thread precisa de sua
própria `cl.CommandQueue` (a spec OpenCL define `cl_command_queue` como não thread-safe).
A solução deve ser transparente: nenhum arquivo fora de `opencl_context.py` pode mudar.

## Decisão

`VglClContext.commandQueue` vira uma `@property` que delega para
`opencl_context.queue` (também uma `@property` com `threading.local()`).
`get_vglClContext_attributes()` passa `self` (o objeto `opencl_context`) ao `VglClContext`
em vez da fila estática.

## Alternativas Consideradas

1. **Lock global em torno da fila única**
   Envolve cada `enqueue_*` com `threading.Lock()`. Elimina concorrência na queue.
   Rejeitada: serializa todos os acessos, impedindo o paralelismo real que é o objetivo.

2. **`VglClContext` por thread via `threading.local()` no caller**
   Cada thread chamaria `get_vglClContext_attributes()` para obter seu próprio contexto.
   Rejeitada: exige modificar todos os callers (`vglClUtil.py`, `vglClImage.py`, etc.).

## Consequências

- ✅ Zero mudanças fora de `opencl_context.py`
- ✅ Retrocompatível: callers existentes não percebem a mudança
- ✅ Cada thread paralela recebe sua própria fila automaticamente
- ⚠️ `VglClContext` deixa de ser um DTO simples — passa a depender de `opencl_context`
