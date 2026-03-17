---
id: techspec-cliente-servidor
prd: prd-cliente-servidor
status: draft
created: 2026-03-17
complexity: Large
---

# TechSpec — Arquitetura Cliente-Servidor VGLGui

## Resumo Executivo

**Feature:** Servidor de execução FastAPI + refatoração do `runner.py` como cliente HTTP/WS
**Complexidade:** Large
**Componentes:** 1 novo (`executor_server.py`) + 2 modificados (`runner.py`, `app.py`) + 1 levemente modificado (`execWorkflow.py`)

**Decisões Principais:**
1. Servidor FastAPI + uvicorn rodando como **subprocess separado** do processo da GUI — evita conflito com Dear PyGui e isola falhas
2. Executor (`execWorkflow.py`) continua como subprocess do **servidor** — preserva isolamento OpenCL
3. Sinal de skip de glyph via **arquivo temporário** — IPC simples sem mudar assinatura do executor
4. Buffer de eventos em memória (list) por job — replay ao conectar WS late
5. Parsing de stdout do executor → eventos JSON no servidor (herda lógica existente do `runner.py`)

**Critérios de verificação:** 14 definidos
**Riscos críticos:** 2
**Dependências bloqueantes:** nenhuma (fastapi + uvicorn instaláveis via pip)

---

## 1. Background

O `runner.py` atual sobe `execWorkflow.py` como subprocess, lê stdout linha a linha via regex e atualiza a GUI. Todo o estado de execução vive no processo da GUI. Não há canal de controle bidirecional.

A dissertação (Maciel 2021, Fig. 6) define as camadas 3 (Interpretation = Workflow Interpreter + Web Server) e 4 (Communication = Request/Communication Module) como responsabilidades do servidor e do módulo de comunicação do cliente, respectivamente. Esta TechSpec implementa exatamente essas duas camadas adaptadas para o stack Python/Dear PyGui.

---

## 2. Arquitetura Geral

```
┌─────────────────────────────────────────────┐
│  Processo GUI (Dear PyGui)                  │
│                                             │
│  app.py ──spawn──► executor_server (proc)   │
│                         ▲                   │
│  runner.py  ──HTTP/WS───┘                   │
│  (camada 4 / Communication)                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Processo executor_server.py (FastAPI :8765)│
│  (camada 3 / Interpretation)                │
│                                             │
│  JobManager ──spawn──► execWorkflow.py      │
│                              │ stdout       │
│                         EventParser         │
│                              │ events       │
│                         EventBus ──WS──► GUI│
└─────────────────────────────────────────────┘
```

**Fluxo completo de uma execução:**

```
GUI clica Run
  → runner.run_workflow("GPU")
      → POST /run  {wksp_content, device}
          → servidor salva wksp em tmp
          → servidor spawna execWorkflow.py
          → retorna {job_id}
      → runner conecta WS /events/{job_id}
      → runner processa eventos:
          glyph_start → queue_glyph_status(gid, "running")
          glyph_done  → queue_glyph_status(gid, "done")
          log         → append_log(line)
          show_image  → queue_show_image(path)
          finished    → APP_STATE["exec_running"] = False

GUI clica Stop
  → runner.stop_workflow()
      → POST /stop/{job_id}
          → servidor termina subprocess
          → emite evento {"type":"stopped"}

GUI clica Skip Glyph (futuro botão)
  → POST /stop/{job_id}/glyph/{glyph_id}
      → servidor cria arquivo /tmp/vgl_skip_{job_id}_{glyph_id}
      → execWorkflow.py checa arquivo no início de cada glyph
```

---

## 3. Design dos Componentes

---

### 3.1 `executor_server.py` (NOVO)

**Localização:** `executor_server.py` (raiz do projeto, ao lado de `execWorkflow.py`)

#### 3.1.1 Modelos de dados

```python
from dataclasses import dataclass, field
from typing import Optional
import asyncio

@dataclass
class JobState:
    job_id: str
    device: str
    status: str          # "running" | "done" | "error" | "stopped"
    process: object      # subprocess.Popen — None após término
    events: list         # buffer de todos os eventos (replay)
    ws_clients: list     # lista de asyncio.Queue ativas
    glyph_status: dict   # glyph_id → "running"|"done"|"error"|"skipped"
    current_glyph: Optional[str]
    elapsed: float
    wksp_tmp: str        # caminho do arquivo .wksp temporário
```

#### 3.1.2 Estado global do servidor

```python
_job: Optional[JobState] = None   # apenas 1 job por vez (v1)
_job_lock = asyncio.Lock()        # protege acesso a _job
```

#### 3.1.3 Endpoints REST

```python
# GET /health
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# POST /run
# Body: {"wksp_content": str, "device": "GPU"|"CPU"}
# Retorna: {"job_id": str, "status": "running"} HTTP 202
# Erro: {"error": "job_running", "job_id": str} HTTP 409

# GET /status/{job_id}
# Retorna: {status, device, elapsed, current_glyph, glyphs: {gid: status}}
# Erro: HTTP 404

# POST /stop/{job_id}
# Termina job (proc.terminate())
# Erro: HTTP 404 | 409 (já terminou)

# POST /stop/{job_id}/glyph/{glyph_id}
# Cria arquivo de sinal de skip
# Erro: HTTP 404
```

#### 3.1.4 WebSocket `/events/{job_id}`

```python
@app.websocket("/events/{job_id}")
async def ws_events(websocket: WebSocket, job_id: str):
    await websocket.accept()

    # Replay buffer de eventos anteriores
    for event in _job.events:
        await websocket.send_json(event)

    # Subscreve na fila de novos eventos
    queue = asyncio.Queue()
    _job.ws_clients.append(queue)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["type"] in ("finished", "stopped", "error"):
                break
    finally:
        _job.ws_clients.remove(queue)
```

#### 3.1.5 JobManager — spawn e leitura de stdout

```python
async def _run_job(job: JobState):
    """Roda em asyncio task. Spawna execWorkflow.py e parseia stdout."""
    env = _build_env(job.device)
    proc = subprocess.Popen(
        [sys.executable, "execWorkflow.py", job.wksp_tmp, job.device],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
    )
    job.process = proc

    # Leitura assíncrona de stdout
    loop = asyncio.get_event_loop()
    executed = set()

    async for line in _iter_stdout(proc):
        # [GUI_SHOW] → evento show_image
        m = _SHOW_IMG_RE.search(line)
        if m:
            await _emit(job, {"type": "show_image",
                               "glyph_id": job.current_glyph,
                               "path": m.group(1).strip()})
            continue

        # Linha de log normal
        await _emit(job, {"type": "log", "line": line})

        # Detecta início de função
        m = _FUNC_START_RE.search(line)
        if m:
            if job.current_glyph:
                await _emit(job, {"type": "glyph_done",
                                   "glyph_id": job.current_glyph,
                                   "status": "done"})
                executed.add(job.current_glyph)

            func = m.group(1)
            gid = _find_glyph(func, executed)
            job.current_glyph = gid
            if gid:
                job.glyph_status[gid] = "running"
                await _emit(job, {"type": "glyph_start",
                                   "glyph_id": gid, "func": func})

    proc.wait()
    rc = proc.returncode

    if job.current_glyph:
        status = "done" if rc == 0 else "error"
        await _emit(job, {"type": "glyph_done",
                           "glyph_id": job.current_glyph,
                           "status": status})

    job.status = "done" if rc == 0 else "error"
    await _emit(job, {"type": "finished", "returncode": rc,
                       "elapsed": job.elapsed})

    _cleanup_job(job)
```

#### 3.1.6 Emissão de eventos

```python
async def _emit(job: JobState, event: dict):
    """Adiciona timestamp, guarda no buffer, distribui para clientes WS."""
    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
    job.events.append(event)

    # Atualiza estado interno
    if event["type"] == "glyph_start":
        job.glyph_status[event["glyph_id"]] = "running"
        job.current_glyph = event["glyph_id"]
    elif event["type"] == "glyph_done":
        job.glyph_status[event["glyph_id"]] = event["status"]

    # Distribui para todos os clientes WebSocket conectados
    for q in list(job.ws_clients):
        await q.put(event)
```

#### 3.1.7 Sinal de skip (RF-006)

```python
def _skip_signal_path(job_id: str, glyph_id: str) -> str:
    return f"/tmp/vgl_skip_{job_id}_{glyph_id}"

# POST /stop/{job_id}/glyph/{glyph_id}
async def stop_glyph(job_id: str, glyph_id: str):
    open(_skip_signal_path(job_id, glyph_id), "w").close()  # cria arquivo vazio
    return {"ok": True}
```

#### 3.1.8 Entry point

```python
if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
```

---

### 3.2 `execWorkflow.py` (MODIFICADO — mínimo)

Adicionar **checagem de sinal de skip** no início de cada handler de glyph. A mudança é local: uma função utilitária + chamada no início do loop.

```python
# Adicionar perto do topo (após imports existentes)
def _check_skip(glyph_id: str) -> bool:
    """Retorna True se o glyph deve ser pulado (sinal externo)."""
    job_id = os.environ.get("VGL_JOB_ID", "")
    if not job_id:
        return False
    sig = f"/tmp/vgl_skip_{job_id}_{glyph_id}"
    if os.path.exists(sig):
        print(f"[SKIP] Glyph {glyph_id} pulado por sinal externo")
        try:
            os.unlink(sig)
        except OSError:
            pass
        return True
    return False
```

No loop `for vGlyph in workspace.lstGlyph:`, adicionar no início:

```python
for vGlyph in workspace.lstGlyph:
    if vGlyph.glyph_id in processed_workflows:
        continue
    if _check_skip(vGlyph.glyph_id):      # ← NOVO
        GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)
        continue
    # ... resto do código sem alteração
```

O servidor passa `VGL_JOB_ID` como variável de ambiente ao spawnar o subprocess.

---

### 3.3 `gui/runner.py` (REFATORADO)

O `runner.py` torna-se um cliente HTTP/WS. As assinaturas públicas `run_workflow()` e `stop_workflow()` não mudam — a GUI não percebe diferença.

```python
import httpx          # cliente HTTP síncrono (já disponível ou via pip)
import websockets     # cliente WS assíncrono
import threading
import asyncio
import json

_SERVER_URL = "http://127.0.0.1:8765"
_WS_URL     = "ws://127.0.0.1:8765"
_current_job_id: str | None = None


def run_workflow(device: str = "GPU"):
    if APP_STATE["exec_running"]:
        append_log(t("runner_running"))
        return
    if not APP_STATE["glyphs"]:
        append_log(t("runner_empty"))
        return

    warnings = _validate_workflow()
    for w in warnings:
        append_log(w)
    if any(w.startswith("[ERRO]") for w in warnings):
        return

    # Serializa o workspace atual
    from gui.wksp_io import save_wksp_to_string
    wksp_content = save_wksp_to_string()

    # Reseta status visual
    from gui.canvas import set_glyph_status
    for gid in APP_STATE["glyphs"]:
        set_glyph_status(gid, "ready")

    APP_STATE["exec_running"] = True
    append_log(t("runner_starting", device=device))

    threading.Thread(target=_run_via_server,
                     args=(wksp_content, device), daemon=True).start()


def stop_workflow():
    global _current_job_id
    if _current_job_id and APP_STATE["exec_running"]:
        try:
            httpx.post(f"{_SERVER_URL}/stop/{_current_job_id}", timeout=3)
        except Exception:
            pass
        append_log(t("runner_stopped"))


def _run_via_server(wksp_content: str, device: str):
    global _current_job_id
    import time as _time

    try:
        resp = httpx.post(f"{_SERVER_URL}/run",
                          json={"wksp_content": wksp_content, "device": device},
                          timeout=5)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        _current_job_id = job_id
    except Exception as e:
        append_log(t("runner_error", err=e))
        APP_STATE["exec_running"] = False
        return

    # Consome eventos WebSocket em loop síncrono via asyncio
    asyncio.run(_consume_events(job_id))


async def _consume_events(job_id: str):
    from gui.canvas import queue_glyph_status, queue_node_preview
    from gui.image_preview import queue_show_image
    import time as _time

    t0 = _time.time()
    executed: set[str] = set()

    async with websockets.connect(f"{_WS_URL}/events/{job_id}") as ws:
        async for raw in ws:
            event = json.loads(raw)
            etype = event["type"]

            if etype == "glyph_start":
                queue_glyph_status(event["glyph_id"], "running")

            elif etype == "glyph_done":
                gid = event["glyph_id"]
                s = event["status"]  # "done" | "error" | "skipped"
                state = APP_STATE["glyphs"].get(gid)
                final = ("ready" if state and state.func == "vglCreateImage"
                         else s)
                queue_glyph_status(gid, final)
                executed.add(gid)

            elif etype == "log":
                append_log(event["line"])

            elif etype == "show_image":
                queue_show_image(event["path"])
                gid = event.get("glyph_id")
                if gid:
                    queue_node_preview(gid, event["path"])

            elif etype in ("finished", "stopped", "error"):
                APP_STATE["exec_running"] = False
                APP_STATE["exec_process"] = None
                elapsed = event.get("elapsed", _time.time() - t0)
                rc = event.get("returncode", -1)
                label = t("runner_ok") if rc == 0 else t("runner_err_code", code=rc)
                append_log(t("runner_finished", label=label))
                append_log(t("runner_time", secs=f"{elapsed:.2f}"))
                break
```

**Nota:** `save_wksp_to_string()` é uma função nova (1 linha) em `wksp_io.py` que usa `io.StringIO` em vez de abrir arquivo — ver §3.5.

---

### 3.4 `gui/app.py` (MODIFICADO)

Adicionar start/stop do servidor no ciclo de vida da GUI.

```python
# Adicionar imports
import subprocess as _sp
import httpx as _httpx
import atexit as _atexit
import time as _time

_server_proc: _sp.Popen | None = None

def _start_server() -> bool:
    """Sobe executor_server.py e aguarda /health. Retorna True se OK."""
    global _server_proc
    server_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "executor_server.py"
    )
    _server_proc = _sp.Popen(
        [sys.executable, server_script, "8765"],
        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
    )
    _atexit.register(_stop_server)

    # Aguarda até 5 segundos
    for _ in range(50):
        _time.sleep(0.1)
        try:
            r = _httpx.get("http://127.0.0.1:8765/health", timeout=0.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

def _stop_server():
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc.wait(timeout=3)
```

Em `run()`, antes de `dpg.create_context()`:

```python
def run():
    _cfg = load_config()
    set_language(_cfg.get("language", "pt"))

    if not _start_server():              # ← NOVO
        print("[ERRO] Servidor de execução não respondeu. Run desabilitado.")
        APP_STATE["server_ok"] = False
    else:
        APP_STATE["server_ok"] = True

    # ... resto sem alteração
```

---

### 3.5 `gui/wksp_io.py` (MODIFICADO — mínimo)

Adicionar `save_wksp_to_string()` reutilizando a lógica existente de `save_wksp()`.

```python
import io

def save_wksp_to_string() -> str:
    """Serializa o workspace atual para string (sem gravar arquivo)."""
    buf = io.StringIO()
    _write_wksp(buf)   # refatorar save_wksp para aceitar file-like object
    return buf.getvalue()
```

A refatoração de `save_wksp()` é extrair o corpo para `_write_wksp(f)` e chamar de ambas as funções.

---

## 4. ADRs (Decisões Arquiteturais)

### ADR-001: FastAPI + uvicorn vs Flask + threads

**Decisão:** FastAPI + uvicorn

**Motivo:** FastAPI tem suporte nativo a WebSocket e async — necessário para streaming de eventos sem bloquear. Flask exigiria Flask-SocketIO + eventlet/gevent, adicionando complexidade. FastAPI é mais simples para o caso de uso específico.

**Consequência:** Dependências `fastapi` e `uvicorn[standard]` adicionadas.

---

### ADR-002: Servidor como processo separado vs thread

**Decisão:** Processo separado (`subprocess.Popen`)

**Motivo:** Dear PyGui **não é thread-safe** e usa seu próprio loop de renderização. uvicorn precisa de um event loop asyncio dedicado. Misturar os dois em threads no mesmo processo cria condições de corrida. Processo separado dá isolamento total: falha no servidor não trava a GUI.

**Consequência:** Comunicação via HTTP/WS em vez de chamadas diretas. Latência negligenciável (loopback).

---

### ADR-003: execWorkflow.py como subprocess do servidor vs inline

**Decisão:** Subprocess do servidor

**Motivo:** `execWorkflow.py` chama `vl.vglClInit()` no nível do módulo — inicializa OpenCL globalmente. Se importado inline, o contexto OpenCL ficaria no processo do servidor, impossibilitando reinicialização entre jobs. Subprocess garante contexto OpenCL limpo a cada execução.

**Consequência:** Parsing de stdout continua necessário. Sinal de skip via arquivo.

---

### ADR-004: Skip de glyph via arquivo temporário vs stdin/pipe

**Decisão:** Arquivo temporário (`/tmp/vgl_skip_{job_id}_{glyph_id}`)

**Motivo:** stdin do subprocess já está fechado (seria necessário manter pipe aberto e modificar execWorkflow.py mais profundamente). Arquivo é a solução mais simples: checagem não bloqueante, zero dependências, fácil de debugar.

**Consequência:** Diretório `/tmp` deve ser gravável (padrão em Linux). Arquivos são deletados após leitura e no `_cleanup_job()`.

---

### ADR-005: httpx vs requests para o cliente HTTP

**Decisão:** `httpx`

**Motivo:** `httpx` é compatível com async e sync, tem interface idêntica ao `requests`, e já é dependência transitiva do Dear PyGui/FastAPI em muitos ambientes. Evita instalar dois clientes HTTP.

**Alternativa rejeitada:** `requests` — não tem cliente async, inviabilizaria eventual migração para cliente 100% async.

---

## 5. Riscos

### RISCO-001: Porta 8765 ocupada no ambiente do usuário
**Probabilidade:** Baixa
**Impacto:** Servidor não sobe; GUI desabilita Run
**Mitigação:** Servidor tenta portas 8765–8775 em sequência; GUI loga qual porta está sendo usada

### RISCO-002: execWorkflow.py com `vl.vglClInit()` no nível do módulo
**Probabilidade:** Confirmada (código atual)
**Impacto:** Se importado diretamente no servidor, contexto OpenCL não reinicializa entre jobs
**Mitigação:** Manter execução como subprocess — já coberto pelo ADR-003

---

## 6. Critérios de Verificação

| RF | Critério | Tipo |
|----|----------|------|
| RF-001 | `GET /health` retorna 200 após start do servidor | Funcional |
| RF-001 | `APP_STATE["server_ok"] == True` após `_start_server()` | Funcional |
| RF-002 | `POST /run` retorna 202 + job_id UUID válido | Funcional |
| RF-002 | Segundo `POST /run` com job ativo retorna 409 | Funcional |
| RF-003 | WS `/events/{job_id}` recebe evento `glyph_start` durante execução | Funcional |
| RF-003 | Conectar ao WS após job iniciado recebe replay de todos os eventos anteriores | Funcional |
| RF-004 | `GET /status/{job_id}` retorna `glyphs` com status de cada nó | Funcional |
| RF-005 | `POST /stop/{job_id}` termina o subprocess e emite evento `stopped` | Funcional |
| RF-005 | Após stop, novo `POST /run` funciona sem reiniciar o servidor | Funcional |
| RF-006 | `POST /stop/{job_id}/glyph/{glyph_id}` cria arquivo de sinal | Funcional |
| RF-006 | Glyph com arquivo de sinal é pulado; emite `glyph_done status=skipped` | Funcional |
| RF-007 | `run_workflow()` e `stop_workflow()` mantêm assinaturas inalteradas | Estrutural |
| RF-007 | Output visual de `demo.wksp` idêntico antes/depois da refatoração | Regressão |
| RNF-002 | Tempo de execução de `fundus.wksp` não aumenta mais de 5% | Performance |

### Script de verificação rápida

```bash
# 1. Sobe servidor manualmente
python executor_server.py 8765 &
sleep 1

# 2. Health
curl -s http://localhost:8765/health
# → {"status":"ok"}

# 3. Run
JOB=$(curl -s -X POST http://localhost:8765/run \
  -H "Content-Type: application/json" \
  -d "{\"wksp_content\": \"$(cat exemplos/demo.wksp | python -c 'import sys,json; print(json.dumps(sys.stdin.read()))')\", \"device\": \"CPU\"}" \
  | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB"

# 4. Status
curl -s http://localhost:8765/status/$JOB | python -m json.tool

# 5. Encerra servidor
kill %1
```

---

## 7. Ordem de Implementação

| Passo | Arquivo | O que fazer | Bloqueia |
|-------|---------|-------------|---------|
| 1 | `requirements.txt` (criar) | adicionar `fastapi`, `uvicorn[standard]`, `httpx`, `websockets` | — |
| 2 | `executor_server.py` | implementar servidor completo (health, run, stop, status, WS) | — |
| 3 | `execWorkflow.py` | adicionar `_check_skip()` + checagem no loop | — |
| 4 | `gui/wksp_io.py` | extrair `_write_wksp()` + adicionar `save_wksp_to_string()` | passo 5 |
| 5 | `gui/runner.py` | refatorar como cliente HTTP/WS | passos 2, 4 |
| 6 | `gui/app.py` | adicionar `_start_server()` / `_stop_server()` | passos 2, 5 |
| 7 | Verificação | rodar script de verificação + `demo.wksp` + `fundus.wksp` | todos |

---

## 8. Rollout

| Aspecto | Estratégia |
|---------|-----------|
| Rollback | `git revert` — mudanças concentradas em 4 arquivos |
| Validação | `demo.wksp` e `fundus.wksp` devem produzir saídas binárias idênticas ao baseline |
| Feature flag | `APP_STATE["server_ok"]` — se False, GUI pode exibir aviso e desabilitar Run |

---

## 9. Questões em Aberto

| # | Questão | Default | Bloqueante? |
|---|---------|---------|------------|
| 1 | `httpx` já está disponível no venv `my_env/`? | Não — instalar | Não |
| 2 | `websockets` já está disponível no venv? | Provavelmente sim (pyopencl pode depender) | Não |
| 3 | O botão "Skip Glyph" (RF-006) vai ficar onde na GUI? | Por ora não há botão — RF-006 disponível via API, UI fica para v2 | Não |

---

## Self-Audit

| Requisito | Coberto? | Seção | AC testável? | Critério? |
|-----------|----------|-------|-------------|---------|
| RF-001 | ✅ | §3.1, §3.4 | ✅ | ✅ §6 |
| RF-002 | ✅ | §3.1.3 | ✅ | ✅ §6 |
| RF-003 | ✅ | §3.1.4, §3.1.5 | ✅ | ✅ §6 |
| RF-004 | ✅ | §3.1.3 | ✅ | ✅ §6 |
| RF-005 | ✅ | §3.1.3 | ✅ | ✅ §6 |
| RF-006 | ✅ | §3.1.7, §3.2 | ✅ | ✅ §6 |
| RF-007 | ✅ | §3.3 | ✅ | ✅ §6 |
| RF-008 | ✅ | §3.4 | ✅ | ✅ §6 |
| RNF-001 | ✅ | ADR-002 (loopback) | ⚠️ manual | — |
| RNF-002 | ✅ | §6 | ✅ | ✅ §6 |
| RNF-003 | ✅ | §7 passo 1 | ✅ | ✅ §7 |
| RNF-004 | ✅ | ADR-003 | ✅ | ✅ §6 |
| RNF-005 | ✅ | §3.4 (`server_ok`) | ✅ | ✅ §6 |
