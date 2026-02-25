import os
import re
import sys
import tempfile
import threading
import subprocess

from gui.app import APP_STATE
from gui.log_panel import append_log

# Regex que detecta o início da execução de um Glyph no stdout do executor
# Padrão atual: "A função vglClConvolution está sendo executada"
_FUNC_START_RE = re.compile(r"A função (\S+) está sendo executada")


def run_workflow(device: str = "GPU"):
    if APP_STATE["exec_running"]:
        append_log("[runner] Já há uma execução em andamento.")
        return

    if not APP_STATE["glyphs"]:
        append_log("[runner] Canvas vazio — nada para executar.")
        return

    # 1. Salva estado atual em arquivo temporário
    tmp = tempfile.NamedTemporaryFile(suffix=".wksp", delete=False, mode="w")
    tmp.close()

    from gui.wksp_io import save_wksp
    save_wksp(tmp.name)

    # 2. Reseta status visual de todos os Glyphs
    from gui.canvas import set_glyph_status
    for gid in APP_STATE["glyphs"]:
        set_glyph_status(gid, "ready")

    APP_STATE["exec_running"] = True
    APP_STATE["device"] = device
    append_log(f"\n[runner] Iniciando execução no {device}...\n")

    # 3. Sobe execução em thread daemon
    def _run():
        env = _build_env(device)
        executor = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "execWorkflow.py",
        )

        try:
            proc = subprocess.Popen(
                [sys.executable, executor, tmp.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.path.dirname(os.path.dirname(__file__)),
            )
        except FileNotFoundError as e:
            append_log(f"[runner] ERRO ao iniciar processo: {e}")
            APP_STATE["exec_running"] = False
            return

        APP_STATE["exec_process"] = proc

        current_gid   = None
        executed_gids = set()

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            append_log(line)

            m = _FUNC_START_RE.search(line)
            if m:
                # Marca glyph anterior como DONE
                if current_gid:
                    from gui.canvas import set_glyph_status
                    set_glyph_status(current_gid, "done")
                    executed_gids.add(current_gid)

                func_name = m.group(1)
                current_gid = _find_next_glyph_by_func(func_name, executed_gids)
                if current_gid:
                    from gui.canvas import set_glyph_status
                    set_glyph_status(current_gid, "running")

        proc.wait()
        final_status = "done" if proc.returncode == 0 else "error"

        # Marca último glyph
        if current_gid:
            from gui.canvas import set_glyph_status
            set_glyph_status(current_gid, final_status)

        APP_STATE["exec_running"] = False
        APP_STATE["exec_process"] = None

        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        label = "OK" if proc.returncode == 0 else f"ERRO (código {proc.returncode})"
        append_log(f"\n[runner] Execução finalizada: {label}\n")

    threading.Thread(target=_run, daemon=True).start()


def stop_workflow():
    proc = APP_STATE.get("exec_process")
    if proc and APP_STATE["exec_running"]:
        proc.terminate()
        APP_STATE["exec_running"] = False
        APP_STATE["exec_process"] = None
        append_log("[runner] Execução interrompida pelo usuário.")


def _find_next_glyph_by_func(func_name: str, already_done: set) -> str | None:
    """Retorna o glyph_id do próximo Glyph com func == func_name ainda não executado."""
    for gid, state in APP_STATE["glyphs"].items():
        if state.func == func_name and gid not in already_done:
            return gid
    return None


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
