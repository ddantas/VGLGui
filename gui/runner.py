import os
import re
import sys
import tempfile
import threading
import subprocess

from gui.app import APP_STATE
from gui.log_panel import append_log
from gui.i18n import t

# Regex que detecta o início da execução de um Glyph no stdout do executor
_FUNC_START_RE = re.compile(r"A função (\S+) está sendo executada")
# Regex que detecta caminho de imagem emitido pelo ShowImage
_SHOW_IMG_RE   = re.compile(r"\[GUI_SHOW\] (.+)")


def run_workflow(device: str = "GPU"):
    if APP_STATE["exec_running"]:
        append_log(t("runner_running"))
        return

    if not APP_STATE["glyphs"]:
        append_log(t("runner_empty"))
        return

    # Validação prévia
    warnings = _validate_workflow()
    for w in warnings:
        append_log(w)
    if any(w.startswith("[ERRO]") for w in warnings):
        return

    # 1. Salva estado atual em arquivo temporário
    tmp = tempfile.NamedTemporaryFile(suffix=".wksp", delete=False, mode="w")
    tmp.close()

    from gui.wksp_io import save_wksp
    save_wksp(tmp.name)

    # 2. Reseta status visual de todos os Glyphs (main thread — ok)
    from gui.canvas import set_glyph_status, queue_glyph_status
    for gid in APP_STATE["glyphs"]:
        set_glyph_status(gid, "ready")

    APP_STATE["exec_running"] = True
    APP_STATE["device"] = device
    append_log(t("runner_starting", device=device))

    # 3. Sobe execução em thread daemon
    def _run():
        import time as _time
        _t0 = _time.time()
        env = _build_env(device)
        executor = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "execWorkflow.py",
        )

        try:
            proc = subprocess.Popen(
                [sys.executable, executor, tmp.name, device],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )
        except FileNotFoundError as e:
            append_log(t("runner_error", err=e))
            APP_STATE["exec_running"] = False
            return

        APP_STATE["exec_process"] = proc

        current_gid   = None
        executed_gids = set()

        for raw_line in proc.stdout:
            line = raw_line.rstrip()

            # Não exibe linhas internas [GUI_SHOW] no log
            sm = _SHOW_IMG_RE.search(line)
            if sm:
                img_path = sm.group(1).strip()
                from gui.image_preview import queue_show_image
                queue_show_image(img_path)
                if current_gid:
                    from gui.canvas import queue_node_preview
                    queue_node_preview(current_gid, img_path)
                continue

            append_log(line)

            m = _FUNC_START_RE.search(line)
            if m:
                # Marca glyph anterior como concluído.
                # vglCreateImage volta para "ready" (cinza) — é só alocação.
                if current_gid:
                    prev_state = APP_STATE["glyphs"].get(current_gid)
                    done_status = ("ready"
                                  if prev_state and prev_state.func == "vglCreateImage"
                                  else "done")
                    queue_glyph_status(current_gid, done_status)
                    executed_gids.add(current_gid)

                func_name = m.group(1)
                current_gid = _find_next_glyph_by_func(func_name, executed_gids)
                if current_gid:
                    queue_glyph_status(current_gid, "running")

        proc.wait()
        final_status = "done" if proc.returncode == 0 else "error"

        # Marca último glyph
        if current_gid:
            prev_state = APP_STATE["glyphs"].get(current_gid)
            last_status = ("ready"
                           if prev_state and prev_state.func == "vglCreateImage"
                           else final_status)
            queue_glyph_status(current_gid, last_status)

        APP_STATE["exec_running"] = False
        APP_STATE["exec_process"] = None

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        elapsed = _time.time() - _t0
        label = (t("runner_ok") if proc.returncode == 0
                 else t("runner_err_code", code=proc.returncode))
        append_log(t("runner_finished", label=label))
        append_log(t("runner_time", secs=f"{elapsed:.2f}"))

    threading.Thread(target=_run, daemon=True).start()


def stop_workflow():
    proc = APP_STATE.get("exec_process")
    if proc and APP_STATE["exec_running"]:
        proc.terminate()
        APP_STATE["exec_running"] = False
        APP_STATE["exec_process"] = None
        append_log(t("runner_stopped"))


def _find_next_glyph_by_func(func_name: str, already_done: set) -> str | None:
    """Retorna o glyph_id do próximo Glyph com func == func_name ainda não executado."""
    for gid, state in APP_STATE["glyphs"].items():
        if state.func == func_name and gid not in already_done:
            return gid
    return None


def _validate_workflow() -> list[str]:
    """Verifica problemas comuns antes de executar. Retorna lista de avisos/erros."""
    from gui.canvas import get_attr_tag_to_port
    from gui.glyph_registry import GLYPH_REGISTRY

    attr_map = get_attr_tag_to_port()
    msgs = []

    # Portas conectadas: conjunto de (glyph_id, port_key)
    connected_inputs: set[tuple[str, str]] = set()
    for out_attr, in_attr in APP_STATE["links"].values():
        in_info = attr_map.get(in_attr)
        if in_info:
            connected_inputs.add(in_info)  # (glyph_id, port_key)

    for gid, state in APP_STATE["glyphs"].items():
        glyph_def = GLYPH_REGISTRY.get(state.func)
        if not glyph_def:
            continue

        # Verifica portas de entrada obrigatórias desconectadas
        for port in glyph_def.ports:
            if port.kind == "input" and port.required:
                port_key = port.name + "_in"
                if (gid, port_key) not in connected_inputs:
                    msgs.append(
                        f"[AVISO] Nó [{gid}] {state.func}: porta '{port.name}' obrigatória não conectada"
                    )

        # Verifica parâmetros de arquivo vazios
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


def _build_env(device: str) -> dict:
    env = os.environ.copy()
    if device == "CPU":
        env["LD_LIBRARY_PATH"] = (
            "/opt/AMDAPPSDK-2.9-1/lib/x86_64/:"
            + env.get("LD_LIBRARY_PATH", "")
        )
    else:  # GPU
        env["LD_LIBRARY_PATH"] = (
            "/opt/amdgpu/lib/x86_64-linux-gnu/:/opt/rocm/lib/:"
            + env.get("LD_LIBRARY_PATH", "")
        )
    return env
