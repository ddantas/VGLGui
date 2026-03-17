---
prd: prd-cliente-servidor
techspec: techspec-cliente-servidor
created: 2026-03-17
complexity: Large
total_tasks: 7
---

# Tasks — Arquitetura Cliente-Servidor VGLGui

## Proteções globais (NÃO alterar em nenhuma task)
- `readWorkflow.py`
- `vgl_lib/` (exceto se explicitamente mencionado)
- `gui/canvas.py`, `gui/sidebar_glyphs.py`, `gui/toolbar.py`, `gui/log_panel.py`
- Formato dos arquivos `.wksp` existentes

---

## Fase 1 — Infraestrutura (independentes entre si)

- [ ] **1.1** [RF-001, RNF-003] Criar `requirements.txt` com dependências novas
  → arquivo: `1_1-criar-requirements.md`

- [ ] **1.2** [RF-001, RF-002, RF-003, RF-004, RF-005, RF-006] Implementar `executor_server.py`
  → arquivo: `1_2-implementar-executor-server.md`

- [ ] **1.3** [RF-006] Adicionar `_check_skip()` em `execWorkflow.py`
  → arquivo: `1_3-adicionar-skip-execworkflow.md`

---

## Fase 2 — Integração (dependem da Fase 1)

- [ ] **2.1** [RF-007] Refatorar `gui/wksp_io.py` — extrair `save_wksp_to_string()`
  → arquivo: `2_1-refatorar-wksp-io.md`
  → depende: 1.1

- [ ] **2.2** [RF-007] Refatorar `gui/runner.py` como cliente HTTP/WS
  → arquivo: `2_2-refatorar-runner-cliente.md`
  → depende: 1.2, 2.1

- [ ] **2.3** [RF-008] Integrar start/stop do servidor em `gui/app.py`
  → arquivo: `2_3-integrar-server-app.md`
  → depende: 1.2, 2.2

---

## Fase 3 — Verificação

- [ ] **3.1** [todos RFs + RNFs] Verificação end-to-end
  → arquivo: `3_1-verificacao.md`
  → depende: 2.3

---

## Grafo de dependências

```
1.1 ──────────────────────────────► 2.1 ──► 2.2 ──► 2.3 ──► 3.1
1.2 ──────────────────────────────────────► 2.2
1.3 (independente, usada em runtime pelo servidor)
```

---

## Traceabilidade Reversa

| Requisito | Tasks | Coberto? |
|-----------|-------|----------|
| RF-001 | 1.2, 2.3 | ✅ |
| RF-002 | 1.2 | ✅ |
| RF-003 | 1.2 | ✅ |
| RF-004 | 1.2 | ✅ |
| RF-005 | 1.2, 2.2 | ✅ |
| RF-006 | 1.2, 1.3 | ✅ |
| RF-007 | 2.1, 2.2 | ✅ |
| RF-008 | 2.3 | ✅ |
| RNF-001 | 1.2 (loopback WS) | ✅ |
| RNF-002 | 3.1 (regressão) | ✅ |
| RNF-003 | 1.1 | ✅ |
| RNF-004 | 3.1 (regressão .wksp) | ✅ |
| RNF-005 | 2.3 (server_ok flag) | ✅ |
