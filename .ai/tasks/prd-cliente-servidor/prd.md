---
id: prd-cliente-servidor
title: Arquitetura Cliente-Servidor para Execução de Workflows VGLGui
status: draft
created: 2026-03-17
classification: High
referencia: Maciel, R.W.S. — Dissertação de Mestrado UFS 2021 (Seção 3.1, Tabela 5)
---

# PRD — Arquitetura Cliente-Servidor para Execução de Workflows VGLGui

## 1. Resumo Executivo

O executor atual (`runner.py`) sobe o `execWorkflow.py` como subprocess e lê seu stdout via regex para inferir o estado da execução. Não há canal de controle bidirecional: não é possível parar um glyph específico, inspecionar o estado da execução externamente nem conectar múltiplos clientes ao mesmo job.

Este PRD especifica a implementação da **Camada 3 (Interpretation) + Camada 4 (Communication)** da arquitetura VGLGUI definida na dissertação de referência: um servidor de execução acessível via HTTP/WebSocket, e a adaptação do `runner.py` para se tornar um cliente desse servidor. A feature habilita visibilidade em tempo real, controle granular (parar glyph específico) e separação real entre GUI e executor — abrindo caminho para execução remota em servidores GPU.

---

## 2. Contexto e Motivação

- A **Decisão 7** da Tabela 5 da dissertação (Maciel 2021) estabelece explicitamente: *"Arquitetura da solução com cliente, editor de workflow e servidor."*
- A **Figura 6** da dissertação descreve 6 camadas, onde as camadas 1–3 rodam no servidor e as camadas 4–6 rodam no cliente.
- A **Figura 9** (Diagrama de implantação) mostra: `<<computer>> VGLGUI` se conecta via `workflow` a `<<server>> Images Processing Server`, que agenda para servidores de CPU e GPU.
- O protótipo atual implementou apenas as camadas 1–2 (VisionGL + executor) e a camada 6 (GUI). As camadas 3 (Web Server + Workflow Interpreter) e 4 (Communication Module) estão ausentes.
- O `runner.py` atual usa parsing de stdout como canal de comunicação — frágil, unidirecional e não extensível.

---

## 3. Escopo e Non-Goals

### Em escopo (v1)

- Servidor de execução (`executor_server.py`) com FastAPI + WebSocket
- API REST para iniciar, parar e consultar status de jobs
- Stream de eventos em tempo real via WebSocket (por job)
- Adaptação do `runner.py` como cliente HTTP/WS (mantendo a GUI inalterada)
- Controle de parada por job inteiro
- Controle de parada por glyph específico (pula o glyph atual e continua)
- Suporte a execução local (servidor sobe automaticamente junto com a GUI)

### Non-Goals

| ID | Non-Goal | Motivo |
|----|----------|--------|
| NG-001 | Interface web (substituir Dear PyGui por browser) | A dissertação previa WebGL; o TCC já optou por Dear PyGui |
| NG-002 | Autenticação de usuário | Decisão 6 da dissertação — fora do escopo do TCC v1 |
| NG-003 | Fila de múltiplos jobs simultâneos (FCFS scheduler) | Decisão 8 da dissertação — será v2; v1 aceita 1 job por vez |
| NG-004 | Execução remota em servidor diferente da máquina local | Arquitetura suporta, mas não é testada nem documentada no TCC v1 |
| NG-005 | Alteração do formato `.wksp` | Arquivos existentes continuam funcionando sem modificação |
| NG-006 | Persistência de histórico de jobs em banco de dados | Decisão 1 da dissertação — fora do escopo v1 |
| NG-007 | Paralelismo de ramos (coberto pelo PRD prd-paralelismo) | PRD separado |

---

## 4. Requisitos Funcionais

---

### RF-001 [MUST]: Servidor de Execução

O sistema DEVE fornecer um servidor HTTP/WebSocket (`executor_server.py`) que gerencia a execução de workflows.

**Acceptance Criteria:**
- AC-001-A: O servidor sobe na porta `8765` (configurável) ao iniciar a GUI e termina quando a GUI fecha.
- AC-001-B: O servidor responde `GET /health` com `{"status": "ok"}` quando disponível.
- AC-001-C: Se a porta já estiver em uso, o servidor tenta a próxima porta disponível e a GUI é informada.

---

### RF-002 [MUST]: Iniciar Execução via API

O cliente DEVE poder enviar um workflow para execução e receber um identificador de job.

**Acceptance Criteria:**
- AC-002-A: `POST /run` com body `{"wksp_content": "<conteúdo do .wksp>", "device": "GPU"}` retorna `{"job_id": "<uuid>", "status": "queued"}` com HTTP 202.
- AC-002-B: Se já existe um job em execução, `POST /run` retorna HTTP 409 com `{"error": "job_running", "job_id": "<id do job ativo>"}`.
- AC-002-C: O `job_id` é único por execução (UUID v4).

---

### RF-003 [MUST]: Stream de Eventos em Tempo Real

O cliente DEVE poder receber eventos de execução em tempo real via WebSocket.

**Acceptance Criteria:**
- AC-003-A: `WS /events/{job_id}` emite eventos JSON durante toda a execução do job.
- AC-003-B: Evento de início de glyph: `{"type": "glyph_start", "glyph_id": "g3", "func": "vglClDilate", "timestamp": "..."}`.
- AC-003-C: Evento de conclusão de glyph: `{"type": "glyph_done", "glyph_id": "g3", "status": "done"|"error", "timestamp": "..."}`.
- AC-003-D: Evento de linha de log: `{"type": "log", "line": "...", "timestamp": "..."}`.
- AC-003-E: Evento de imagem gerada: `{"type": "show_image", "glyph_id": "g5", "path": "/tmp/...", "timestamp": "..."}`.
- AC-003-F: Evento de conclusão do job: `{"type": "finished", "returncode": 0, "elapsed": 1.23, "timestamp": "..."}`.
- AC-003-G: Conexão WebSocket recebe todos os eventos perdidos desde o início do job (replay do buffer) ao conectar com o job já em andamento.

---

### RF-004 [MUST]: Consultar Status do Job

O cliente DEVE poder consultar o estado atual de um job sem manter conexão WebSocket.

**Acceptance Criteria:**
- AC-004-A: `GET /status/{job_id}` retorna JSON com: `status` ("queued"|"running"|"done"|"error"|"stopped"), `device`, `elapsed`, `current_glyph`, `glyphs` (dict glyph_id → status).
- AC-004-B: Retorna HTTP 404 se o `job_id` não existe.
- AC-004-C: O endpoint responde em menos de 50ms (não bloqueia a execução).

---

### RF-005 [MUST]: Parar Execução Completa

O cliente DEVE poder interromper um job inteiro.

**Acceptance Criteria:**
- AC-005-A: `POST /stop/{job_id}` termina o processo executor e emite evento `{"type": "stopped", "reason": "user_request"}` no WebSocket.
- AC-005-B: Após `POST /stop`, `GET /status/{job_id}` retorna `status: "stopped"`.
- AC-005-C: O servidor fica disponível para novo `POST /run` imediatamente após o stop.
- AC-005-D: Se o job já terminou, `POST /stop/{job_id}` retorna HTTP 409.

---

### RF-006 [SHOULD]: Parar Glyph Específico

O cliente DEVE poder sinalizar ao executor para pular o glyph em execução e continuar com o próximo.

**Acceptance Criteria:**
- AC-006-A: `POST /stop/{job_id}/glyph/{glyph_id}` sinaliza o executor para pular o glyph especificado.
- AC-006-B: O executor checa o sinal entre glyphs (não interrompe no meio de uma chamada VGL).
- AC-006-C: O glyph pulado emite evento `{"type": "glyph_done", "status": "skipped"}`.
- AC-006-D: A execução continua normalmente a partir do próximo glyph na topologia.
- AC-006-E: Se o `glyph_id` não está no job ativo, retorna HTTP 404.

---

### RF-007 [MUST]: Adaptação do runner.py como Cliente

O `runner.py` DEVE ser refatorado para usar a API do servidor, mantendo a interface com a GUI inalterada.

**Acceptance Criteria:**
- AC-007-A: As funções `run_workflow(device)` e `stop_workflow()` mantêm as mesmas assinaturas.
- AC-007-B: O `APP_STATE` continua sendo atualizado via `queue_glyph_status` e `append_log`, sem mudanças no canvas.
- AC-007-C: A GUI não percebe diferença de comportamento em relação à versão anterior.
- AC-007-D: O `runner.py` conecta ao WebSocket do job e processa eventos no lugar do parsing de stdout.

---

### RF-008 [MUST]: Gerenciamento de Ciclo de Vida do Servidor

A GUI DEVE gerenciar o ciclo de vida do servidor automaticamente.

**Acceptance Criteria:**
- AC-008-A: O servidor é iniciado como subprocess quando `app.py` inicia, antes de montar a UI.
- AC-008-B: O servidor é terminado graciosamente quando a janela da GUI fecha.
- AC-008-C: Se o servidor não responder ao `GET /health` em 5 segundos após o start, a GUI exibe erro no log e desabilita o botão Run.
- AC-008-D: O servidor roda em processo separado (não thread) para não bloquear o GIL Python.

---

## 5. Requisitos Não-Funcionais

| ID | Requisito | Critério |
|----|-----------|---------|
| RNF-001 | Latência de eventos | Eventos WebSocket chegam ao cliente em < 100ms após ocorrerem no executor |
| RNF-002 | Sem regressão de performance | Tempo de execução de workflows existentes não aumenta mais de 5% |
| RNF-003 | Sem novas dependências obrigatórias | `fastapi` e `uvicorn[standard]` adicionados ao `requirements.txt`; `websockets` já no env |
| RNF-004 | Compatibilidade com `.wksp` existentes | Todos os arquivos `.wksp` do repositório continuam executando corretamente |
| RNF-005 | Isolamento de falha | Falha no servidor não trava a GUI; exibe erro no log e permite reiniciar o servidor |

---

## 6. Modelo de Dados dos Eventos WebSocket

```json
// glyph_start
{"type": "glyph_start", "glyph_id": "g3", "func": "vglClDilate", "timestamp": "2026-03-17T10:00:00.123Z"}

// glyph_done
{"type": "glyph_done", "glyph_id": "g3", "status": "done", "timestamp": "..."}

// log
{"type": "log", "line": "Executando vglClDilate...", "timestamp": "..."}

// show_image
{"type": "show_image", "glyph_id": "g5", "path": "/tmp/vgl_preview_abc.png", "timestamp": "..."}

// finished
{"type": "finished", "returncode": 0, "elapsed": 2.41, "timestamp": "..."}

// stopped
{"type": "stopped", "reason": "user_request", "timestamp": "..."}

// error
{"type": "error", "message": "FileNotFoundError: imagem.png", "timestamp": "..."}
```

---

## 7. Arquitetura — Mapeamento com a Dissertação (Maciel 2021)

```
Figura 6 da dissertação        →   Implementação neste PRD
─────────────────────────────────────────────────────────────────
Layer 6 - Client (GUI)         →   gui/ (Dear PyGui) — sem mudança
Layer 5 - Framework            →   gui/app.py, canvas.py etc. — sem mudança
Layer 4 - Communication        →   gui/runner.py (refatorado como cliente HTTP/WS)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
Layer 3 - Interpretation       →   executor_server.py (FastAPI + uvicorn)
Layer 2 - Support              →   execWorkflow.py + vgl_lib/ — sem mudança
Layer 1 - Basic Elements       →   Shaders OpenCL — sem mudança
```

```
Figura 9 (Deployment) adaptada:

<<computer>> VGLGUI                    <<process>> ExecutorServer
  gui/main.py (Dear PyGui)               executor_server.py (FastAPI :8765)
  gui/runner.py (HTTP client)   ──────►  POST /run, WS /events/{id}
                                          │
                                          ▼
                                  execWorkflow.py (subprocess)
                                          │
                                  vgl_lib/ + OpenCL (CPU/GPU)
```

---

## 8. Arquivos Afetados

| Arquivo | Mudança |
|---------|---------|
| `executor_server.py` | **NOVO** — servidor FastAPI |
| `gui/runner.py` | **REFATORAR** — cliente HTTP/WS |
| `gui/app.py` | **MODIFICAR** — start/stop do servidor |
| `requirements.txt` | **MODIFICAR** — adicionar fastapi, uvicorn[standard] |
| `execWorkflow.py` | **MODIFICAR** — adicionar checagem de sinal de skip entre glyphs (RF-006) |

---

## 9. Dependências e Riscos

| Item | Descrição | Mitigação |
|------|-----------|-----------|
| DEP-001 | `fastapi` + `uvicorn` disponíveis no venv | Instalar via pip; já compatíveis com Python 3.10 |
| RISCO-001 | Dear PyGui e uvicorn competindo pelo event loop asyncio | uvicorn roda em processo separado, não thread; sem conflito |
| RISCO-002 | Porta 8765 ocupada | Servidor tenta portas consecutivas (8765–8775) |
| RISCO-003 | Overhead de serialização JSON atrasar eventos | Eventos são pequenos (< 512 bytes); impacto desprezível |
