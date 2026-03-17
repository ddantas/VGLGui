---
prd: prd-threadlocal-queue
techspec: techspec-threadlocal-queue
created: 2026-03-16
complexity: Small
total_tasks: 3
---

# Tasks — CommandQueue Thread-Local

## Proteções globais (NÃO alterar em nenhuma task)
- `vgl_lib/vglClUtil.py`
- `vgl_lib/vglClImage.py`
- `vgl_lib/vglContext.py`
- `vgl_lib/__init__.py`
- `execWorkflow.py` e qualquer caller de `vl.get_ocl().commandQueue`

---

## Fase 1 — Implementação

- [ ] **1.1** [RF-001, RF-002, RF-003] Refatorar `vgl_lib/opencl_context.py` com queue thread-local
  → arquivo: `1_1-refatorar-opencl-context-thread-local.md`

---

## Fase 2 — Verificação

- [ ] **2.1** [P] [RF-001, RF-003] Criar e rodar `test_threadlocal_queue.py`
  → arquivo: `2_1-criar-script-verificacao-thread-local.md`

- [ ] **2.2** [P] [RF-002] Regressão serial — executar workflow e comparar output
  → arquivo: `2_2-verificar-regressao-serial.md`

---

## Traceabilidade Reversa

| Requisito | Tasks | Coberto? |
|---|---|---|
| RF-001 | 1.1, 2.1 | ✅ |
| RF-002 | 1.1, 2.2 | ✅ |
| RF-003 | 1.1, 2.1 | ✅ |
| RNF-001 | 1.1 (log de criação) | ✅ |
| RNF-002 | 2.2 (diff binário) | ✅ |
| RNF-003 | 1.1 (GC Python — sem ação extra) | ✅ |
