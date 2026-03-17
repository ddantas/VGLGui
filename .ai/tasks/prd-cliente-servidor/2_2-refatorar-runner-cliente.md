---
id: 2.2
fase: 2
titulo: Refatorar gui/runner.py como cliente HTTP/WS
requisitos: [RF-005, RF-007]
depende: 1.2, 2.1
paralela: false
---

# 2.2 — Refatorar `gui/runner.py` como cliente HTTP/WS

## Contexto

O `runner.py` atual spawna `execWorkflow.py` diretamente e lê stdout via regex.
Deve ser refatorado para comunicar com o `executor_server.py` via HTTP/WebSocket.

As assinaturas públicas `run_workflow(device)` e `stop_workflow()` NÃO MUDAM.
A GUI (canvas.py, toolbar.py, etc.) não precisa ser alterada.

Ver TechSpec §3.3 para o design completo.

## Arquivos

- **Modificar:** `gui/runner.py`
- **NÃO tocar:** nenhum outro arquivo da GUI

## O que implementar

Substituir o conteúdo de `gui/runner.py` pelo design da TechSpec §3.3, preservando:
- `run_workflow(device: str = "GPU")` — mesma assinatura
- `stop_workflow()` — mesma assinatura
- `_validate_workflow()` — pode ser mantida sem alteração
- `_build_env()` — remover (responsabilidade migra para o servidor)
- `_find_next_glyph_by_func()` — remover (responsabilidade migra para o servidor)

### Estrutura do novo runner.py

```python
import asyncio
import json
import threading

import httpx
import websockets

from gui.app import APP_STATE
from gui.log_panel import append_log
from gui.i18n import t

_SERVER_URL = "http://127.0.0.1:8765"
_WS_URL     = "ws://127.0.0.1:8765"
_current_job_id: str | None = None


def run_workflow(device: str = "GPU"):
    global _current_job_id

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

    from gui.wksp_io import save_wksp_to_string
    wksp_content = save_wksp_to_string()

    from gui.canvas import set_glyph_status
    for gid in APP_STATE["glyphs"]:
        set_glyph_status(gid, "ready")

    APP_STATE["exec_running"] = True
    append_log(t("runner_starting", device=device))

    threading.Thread(
        target=_run_via_server,
        args=(wksp_content, device),
        daemon=True,
    ).start()


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
    try:
        resp = httpx.post(
            f"{_SERVER_URL}/run",
            json={"wksp_content": wksp_content, "device": device},
            timeout=10,
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        _current_job_id = job_id
    except Exception as e:
        append_log(t("runner_error", err=e))
        APP_STATE["exec_running"] = False
        return

    asyncio.run(_consume_events(job_id))


async def _consume_events(job_id: str):
    import time as _time
    from gui.canvas import queue_glyph_status, queue_node_preview
    from gui.image_preview import queue_show_image

    t0 = _time.time()

    async with websockets.connect(f"{_WS_URL}/events/{job_id}") as ws:
        async for raw in ws:
            event = json.loads(raw)
            etype = event["type"]

            if etype == "glyph_start":
                queue_glyph_status(event["glyph_id"], "running")

            elif etype == "glyph_done":
                gid = event["glyph_id"]
                s   = event["status"]   # "done"|"error"|"skipped"
                state = APP_STATE["glyphs"].get(gid)
                final = ("ready"
                         if state and state.func == "vglCreateImage"
                         else s)
                queue_glyph_status(gid, final)

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
                elapsed = event.get("elapsed", round(_time.time() - t0, 2))
                rc      = event.get("returncode", -1)
                label   = (t("runner_ok") if rc == 0
                           else t("runner_err_code", code=rc))
                if etype == "stopped":
                    label = t("runner_stopped")
                append_log(t("runner_finished", label=label))
                append_log(t("runner_time", secs=f"{elapsed:.2f}"))
                break


def _validate_workflow() -> list[str]:
    # Manter idêntico ao runner.py atual
    ...
```

**Atenção:** copiar `_validate_workflow()` do runner.py atual sem alteração.

## Acceptance Criteria

- AC-A: `run_workflow()` e `stop_workflow()` mantêm assinaturas inalteradas
- AC-B: Após `run_workflow()`, o canvas coloriza glyphs normalmente (running/done/error)
- AC-C: `stop_workflow()` emite evento `stopped` e para a execução
- AC-D: `APP_STATE["exec_running"]` volta a `False` após conclusão ou stop
- AC-E: `git diff --name-only` mostra apenas `gui/runner.py`

## Verificar

```bash
git diff --name-only   # deve mostrar só gui/runner.py
```

Teste funcional (requer servidor rodando — task 1.2):
1. Abrir a GUI
2. Carregar `exemplos/demo.wksp`
3. Clicar Run → verificar que glyphs ficam verdes/azuis sequencialmente
4. Carregar workflow longo → clicar Stop → verificar que para
