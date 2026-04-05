import asyncio
import json
import os
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
    APP_STATE["device"] = device
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

    try:
        async with websockets.connect(f"{_WS_URL}/events/{job_id}") as ws:
            async for raw in ws:
                event = json.loads(raw)
                etype = event["type"]

                if etype == "glyph_start":
                    queue_glyph_status(event["glyph_id"], "running")

                elif etype == "glyph_done":
                    gid   = event["glyph_id"]
                    s     = event["status"]   # "done"|"error"|"skipped"
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
                    if etype == "stopped":
                        label = t("runner_stopped")
                    else:
                        label = (t("runner_ok") if rc == 0
                                 else t("runner_err_code", code=rc))
                    append_log(t("runner_finished", label=label))
                    append_log(t("runner_time", secs=f"{elapsed:.2f}"))
                    break
    except Exception as e:
        append_log(t("runner_error", err=e))
        APP_STATE["exec_running"] = False


def _validate_workflow() -> list[str]:
    """Verifica problemas comuns antes de executar. Retorna lista de avisos/erros."""
    from gui.canvas import get_attr_tag_to_port
    from gui.glyph_registry import GLYPH_REGISTRY

    attr_map = get_attr_tag_to_port()
    msgs = []

    connected_inputs: set[tuple[str, str]] = set()
    for out_attr, in_attr in APP_STATE["links"].values():
        in_info = attr_map.get(in_attr)
        if in_info:
            connected_inputs.add(in_info)

    for gid, state in APP_STATE["glyphs"].items():
        glyph_def = GLYPH_REGISTRY.get(state.func)
        if not glyph_def:
            continue

        for port in glyph_def.ports:
            if port.kind == "input" and port.required:
                port_key = port.name + "_in"
                if (gid, port_key) not in connected_inputs:
                    msgs.append(
                        f"[AVISO] Nó [{gid}] {state.func}: porta '{port.name}' obrigatória não conectada"
                    )

        for param in glyph_def.params:
            if param.type == "file":
                val = state.params.get(param.name, "").strip()
                if not val:
                    msgs.append(
                        f"[ERRO] Nó [{gid}] {state.func}: parâmetro '{param.name}' está vazio"
                    )
                elif not os.path.exists(val):
                    msgs.append(
                        f"[AVISO] Nó [{gid}] {state.func}: arquivo não encontrado: {val}"
                    )

    return msgs
