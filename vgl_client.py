#!/usr/bin/env python3
"""
vgl_client.py — Cliente CLI para o executor_server VGLGui

Uso:
  python vgl_client.py status              # estado atual do job
  python vgl_client.py monitor             # stream de eventos em tempo real
  python vgl_client.py stop                # para o job atual
  python vgl_client.py skip <glyph_id>     # pula um glyph específico
  python vgl_client.py run <arquivo.wksp>  # submete um workflow para execução

Opções:
  --server URL   URL base do servidor (default: http://127.0.0.1:8765)
"""
import argparse
import asyncio
import json
import sys

import httpx
import websockets

DEFAULT_SERVER = "http://127.0.0.1:8765"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(server: str, path: str) -> dict:
    try:
        r = httpx.get(f"{server}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        print(f"[ERRO] Servidor não encontrado em {server}. A GUI está aberta?")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"[ERRO] {e.response.status_code}: {e.response.text}")
        sys.exit(1)


def _post(server: str, path: str, body: dict | None = None) -> dict:
    try:
        r = httpx.post(f"{server}{path}", json=body, timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        print(f"[ERRO] Servidor não encontrado em {server}. A GUI está aberta?")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"[ERRO] {e.response.status_code}: {e.response.text}")
        sys.exit(1)


def _current_job(server: str) -> str | None:
    """Retorna o job_id do job ativo, ou None."""
    try:
        data = _get(server, "/current")
        return data.get("job_id")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_status(server: str, args):
    health = _get(server, "/health")
    print(f"Servidor: {health['status'].upper()}  ({server})")

    job_id = args.job_id
    if not job_id:
        data = _get(server, "/current")
        if not data.get("job_id"):
            print("Nenhum job ativo.")
            return
        job_id = data["job_id"]
    else:
        data = _get(server, f"/status/{job_id}")
    status  = data["status"]
    device  = data["device"]
    current = data.get("current_glyph") or "—"
    glyphs  = data.get("glyphs", {})

    icon = {"running": "⏳", "done": "✅", "error": "❌", "stopped": "⛔"}.get(status, "?")
    print(f"\nJob:     {job_id}")
    print(f"Status:  {icon} {status}   Device: {device}")
    print(f"Glyph atual: {current}")

    if glyphs:
        print("\nGlyphs:")
        icons = {"running": "▶", "done": "✓", "error": "✗", "skipped": "⤭", "ready": "·"}
        for gid, st in glyphs.items():
            print(f"  {icons.get(st,'?')} [{gid}] {st}")


def cmd_stop(server: str, args):
    job_id = args.job_id or _current_job(server)
    if not job_id:
        print("Nenhum job ativo para parar.")
        return
    result = _post(server, f"/stop/{job_id}")
    print(f"⛔ Job {job_id} parado." if result.get("ok") else result)


def cmd_skip(server: str, args):
    job_id   = args.job_id or _current_job(server)
    glyph_id = args.glyph_id
    if not job_id:
        print("Nenhum job ativo.")
        return
    result = _post(server, f"/stop/{job_id}/glyph/{glyph_id}")
    if result.get("ok"):
        print(f"⤭  Sinal de skip enviado para glyph [{glyph_id}].")
    else:
        print(result)


def cmd_run(server: str, args):
    import os
    path = args.wksp_file
    if not os.path.isfile(path):
        print(f"[ERRO] Arquivo não encontrado: {path}")
        sys.exit(1)

    content = open(path).read()
    device  = args.device

    result = _post(server, "/run", {"wksp_content": content, "device": device})
    job_id = result.get("job_id")
    if not job_id:
        print(f"[ERRO] {result}")
        sys.exit(1)

    print(f"Job iniciado: {job_id}  (device: {device})")
    print("Monitorando eventos... (Ctrl+C para sair)\n")

    asyncio.run(_stream_events(server, job_id))


async def _stream_events(server: str, job_id: str):
    ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
    try:
        async with websockets.connect(f"{ws_url}/events/{job_id}") as ws:
            async for raw in ws:
                event = json.loads(raw)
                _print_event(event)
                if event["type"] in ("finished", "stopped", "error"):
                    break
    except KeyboardInterrupt:
        print("\n(monitoramento interrompido)")
    except Exception as e:
        print(f"[ERRO WS] {e}")


def cmd_monitor(server: str, args):
    job_id = args.job_id or _current_job(server)
    if not job_id:
        print("Nenhum job ativo para monitorar.")
        return
    print(f"Monitorando job {job_id}... (Ctrl+C para sair)\n")
    asyncio.run(_stream_events(server, job_id))


def _print_event(event: dict):
    etype = event["type"]
    ts    = event.get("timestamp", "")[:19].replace("T", " ")

    if etype == "glyph_start":
        print(f"[{ts}] ▶  [{event['glyph_id']}] {event['func']}")
    elif etype == "glyph_done":
        icons = {"done": "✓", "error": "✗", "skipped": "⤭"}
        icon  = icons.get(event["status"], "?")
        print(f"[{ts}] {icon}  [{event['glyph_id']}] {event['status']}")
    elif etype == "log":
        line = event["line"].strip()
        if line:
            print(f"         {line}")
    elif etype == "show_image":
        print(f"[{ts}] 🖼  {event['path']}")
    elif etype == "finished":
        rc      = event.get("returncode", "?")
        elapsed = event.get("elapsed", "?")
        status  = "✅ OK" if rc == 0 else f"❌ código {rc}"
        print(f"\n[{ts}] {status}  ({elapsed}s)")
    elif etype == "stopped":
        print(f"[{ts}] ⛔ Job parado ({event.get('reason','')})")
    elif etype == "error":
        print(f"[{ts}] ❌ {event.get('message','')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cliente CLI para o executor_server VGLGui",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, metavar="URL",
                        help=f"URL do servidor (default: {DEFAULT_SERVER})")
    parser.add_argument("--job", dest="job_id", default=None, metavar="ID",
                        help="Job ID (necessário para status/stop/skip/monitor)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # status
    sub.add_parser("status", help="Mostra estado do servidor e de um job")

    # monitor
    sub.add_parser("monitor", help="Stream de eventos em tempo real de um job")

    # stop
    sub.add_parser("stop", help="Para o job atual")

    # skip
    p_skip = sub.add_parser("skip", help="Pula um glyph específico")
    p_skip.add_argument("glyph_id", help="ID do glyph a pular (ex: g3)")

    # run
    p_run = sub.add_parser("run", help="Submete um workflow para execução")
    p_run.add_argument("wksp_file", help="Caminho para o arquivo .wksp")
    p_run.add_argument("--device", default="GPU", choices=["GPU", "CPU"],
                       help="Device de execução (default: GPU)")

    args = parser.parse_args()
    server = args.server.rstrip("/")

    cmds = {
        "status":  cmd_status,
        "monitor": cmd_monitor,
        "stop":    cmd_stop,
        "skip":    cmd_skip,
        "run":     cmd_run,
    }
    cmds[args.cmd](server, args)


if __name__ == "__main__":
    main()
