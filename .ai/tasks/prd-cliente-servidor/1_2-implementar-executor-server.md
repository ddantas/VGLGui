---
id: 1.2
fase: 1
titulo: Implementar executor_server.py
requisitos: [RF-001, RF-002, RF-003, RF-004, RF-005, RF-006]
depende: 1.1
paralela: true
---

# 1.2 — Implementar `executor_server.py`

## Contexto

Servidor FastAPI que gerencia o ciclo de vida de jobs de execução de workflow.
Recebe o conteúdo do `.wksp` via HTTP, spawna `execWorkflow.py` como subprocess,
parseia o stdout e distribui eventos JSON via WebSocket.

Ver TechSpec §3.1 para o design completo.

## Arquivos

- **Criar:** `executor_server.py` (raiz do projeto, ao lado de `execWorkflow.py`)
- **NÃO tocar:** nenhum outro arquivo nesta task

## Estrutura completa do arquivo

```python
#!/usr/bin/env python3
"""
executor_server.py — Servidor de execução de workflows VGLGui (Camada 3 / Interpretation)
Referência arquitetural: Maciel 2021, Fig. 6, Camadas 1-3
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Regexes (mesmas do runner.py atual)
# ---------------------------------------------------------------------------
_FUNC_START_RE = re.compile(r"A função (\S+) está sendo executada")
_SHOW_IMG_RE   = re.compile(r"\[GUI_SHOW\] (.+)")

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    wksp_content: str
    device: str = "GPU"

@dataclass
class JobState:
    job_id: str
    device: str
    status: str                              # "running"|"done"|"error"|"stopped"
    process: object                          # subprocess.Popen | None
    events: list = field(default_factory=list)
    ws_queues: list = field(default_factory=list)
    glyph_status: dict = field(default_factory=dict)
    current_glyph: Optional[str] = None
    wksp_tmp: str = ""
    _t0: float = 0.0

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_job: Optional[JobState] = None
_job_lock = asyncio.Lock()

# Mapa func_name → glyph_id (populado ao iniciar o job)
_func_map: dict = {}   # func → [glyph_id, ...]

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="VGLGui Executor Server")

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _build_env(device: str) -> dict:
    env = os.environ.copy()
    if device == "CPU":
        env["LD_LIBRARY_PATH"] = (
            "/opt/AMDAPPSDK-2.9-1/lib/x86_64/:" + env.get("LD_LIBRARY_PATH", "")
        )
    else:
        env["LD_LIBRARY_PATH"] = (
            "/opt/amdgpu/lib/x86_64-linux-gnu/:/opt/rocm/lib/:"
            + env.get("LD_LIBRARY_PATH", "")
        )
    return env

def _skip_path(job_id: str, glyph_id: str) -> str:
    return f"/tmp/vgl_skip_{job_id}_{glyph_id}"

def _cleanup_tmp(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass

async def _emit(job: JobState, event: dict):
    """Adiciona timestamp, guarda no buffer, envia para todos os clientes WS."""
    event.setdefault("timestamp", _now())
    job.events.append(event)

    # Atualiza estado interno
    if event["type"] == "glyph_start":
        job.glyph_status[event["glyph_id"]] = "running"
        job.current_glyph = event["glyph_id"]
    elif event["type"] == "glyph_done":
        job.glyph_status[event["glyph_id"]] = event["status"]

    for q in list(job.ws_queues):
        await q.put(event)

def _find_glyph(func_name: str, executed: set) -> Optional[str]:
    """Retorna o próximo glyph_id com func == func_name ainda não executado."""
    for gid in _func_map.get(func_name, []):
        if gid not in executed:
            return gid
    return None

def _build_func_map(wksp_content: str):
    """Parseia o .wksp para construir func_name → [glyph_ids]."""
    global _func_map
    _func_map = {}
    for line in wksp_content.splitlines():
        if line.startswith("Glyph:"):
            parts = line.split(":")
            # Glyph:library:func::host:id:x:y::
            if len(parts) >= 6:
                func = parts[2]
                gid  = parts[5]
                _func_map.setdefault(func, []).append(gid)

# ---------------------------------------------------------------------------
# Job runner (asyncio task)
# ---------------------------------------------------------------------------
async def _read_stdout_lines(proc):
    """Gerador assíncrono que lê stdout do subprocess linha a linha."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        yield line.rstrip()

async def _run_job(job: JobState):
    global _job
    import time as _time

    job._t0 = _time.time()
    env = _build_env(job.device)
    env["VGL_JOB_ID"] = job.job_id

    executor = os.path.join(os.path.dirname(__file__), "execWorkflow.py")
    try:
        proc = subprocess.Popen(
            [sys.executable, executor, job.wksp_tmp, job.device],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError as e:
        await _emit(job, {"type": "error", "message": str(e)})
        job.status = "error"
        return

    job.process = proc
    executed: set[str] = set()

    async for line in _read_stdout_lines(proc):
        # [GUI_SHOW]
        sm = _SHOW_IMG_RE.search(line)
        if sm:
            await _emit(job, {
                "type": "show_image",
                "glyph_id": job.current_glyph,
                "path": sm.group(1).strip(),
            })
            continue

        # Log normal
        await _emit(job, {"type": "log", "line": line})

        # Detecta início de função
        m = _FUNC_START_RE.search(line)
        if m:
            # Fecha glyph anterior
            if job.current_glyph and job.current_glyph not in executed:
                await _emit(job, {
                    "type": "glyph_done",
                    "glyph_id": job.current_glyph,
                    "status": "done",
                })
                executed.add(job.current_glyph)

            func = m.group(1)
            gid = _find_glyph(func, executed)
            if gid:
                await _emit(job, {"type": "glyph_start", "glyph_id": gid, "func": func})

    await asyncio.get_event_loop().run_in_executor(None, proc.wait)
    rc = proc.returncode
    elapsed = _time.time() - job._t0

    # Fecha último glyph
    if job.current_glyph and job.current_glyph not in executed:
        final_glyph_status = "done" if rc == 0 else "error"
        await _emit(job, {
            "type": "glyph_done",
            "glyph_id": job.current_glyph,
            "status": final_glyph_status,
        })

    job.status = "done" if rc == 0 else ("stopped" if rc == -15 else "error")
    await _emit(job, {
        "type": "finished",
        "returncode": rc,
        "elapsed": round(elapsed, 3),
    })

    _cleanup_tmp(job.wksp_tmp)
    job.process = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run", status_code=202)
async def run(body: RunRequest):
    global _job
    async with _job_lock:
        if _job and _job.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "job_running", "job_id": _job.job_id},
            )

        # Salva wksp em arquivo temporário
        tmp = tempfile.NamedTemporaryFile(suffix=".wksp", delete=False, mode="w")
        tmp.write(body.wksp_content)
        tmp.close()

        _build_func_map(body.wksp_content)

        job_id = str(uuid.uuid4())
        _job = JobState(
            job_id=job_id,
            device=body.device,
            status="running",
            process=None,
            wksp_tmp=tmp.name,
        )

    asyncio.create_task(_run_job(_job))
    return {"job_id": job_id, "status": "running"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job_id,
        "status": _job.status,
        "device": _job.device,
        "current_glyph": _job.current_glyph,
        "glyphs": dict(_job.glyph_status),
        "elapsed": None,  # calculado no evento finished
    }


@app.post("/stop/{job_id}")
async def stop(job_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    if _job.status != "running":
        raise HTTPException(status_code=409, detail="job not running")

    if _job.process and _job.process.poll() is None:
        _job.process.terminate()

    _job.status = "stopped"
    await _emit(_job, {"type": "stopped", "reason": "user_request"})
    return {"ok": True}


@app.post("/stop/{job_id}/glyph/{glyph_id}")
async def stop_glyph(job_id: str, glyph_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")

    # Cria arquivo de sinal de skip
    open(_skip_path(job_id, glyph_id), "w").close()
    return {"ok": True, "glyph_id": glyph_id}


@app.websocket("/events/{job_id}")
async def ws_events(websocket: WebSocket, job_id: str):
    if not _job or _job.job_id != job_id:
        await websocket.close(code=4004)
        return

    await websocket.accept()

    # Replay de eventos anteriores
    for event in list(_job.events):
        await websocket.send_json(event)

    # Se job já terminou, fecha
    if _job.status != "running":
        await websocket.close()
        return

    # Subscreve em novos eventos
    queue: asyncio.Queue = asyncio.Queue()
    _job.ws_queues.append(queue)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["type"] in ("finished", "stopped", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if queue in _job.ws_queues:
            _job.ws_queues.remove(queue)
    await websocket.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
```

## Acceptance Criteria

- AC-A: `GET /health` retorna `{"status": "ok"}` com HTTP 200
- AC-B: `POST /run` com `.wksp` válido retorna HTTP 202 com `job_id` UUID
- AC-C: Segundo `POST /run` enquanto job ativo retorna HTTP 409
- AC-D: `WS /events/{job_id}` recebe eventos durante execução
- AC-E: `POST /stop/{job_id}` termina o processo e emite evento `stopped`
- AC-F: `POST /stop/{job_id}/glyph/{glyph_id}` cria arquivo `/tmp/vgl_skip_*`
- AC-G: Conectar ao WS após job iniciado recebe replay dos eventos anteriores

## Verificar

```bash
cd /home/joao/Documents/TCC_1/VGLGui
source my_env/bin/activate

# Sobe servidor
python executor_server.py 8765 &
sleep 1

# Health
curl -s http://localhost:8765/health
# → {"status":"ok"}

# Segunda chamada /run com job ativo deve dar 409
# (testar após integrar com runner.py na task 2.2)

kill %1
```
