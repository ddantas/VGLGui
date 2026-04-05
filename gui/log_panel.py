import dearpygui.dearpygui as dpg
import threading
from datetime import datetime
from gui.i18n import t

_LOG_CHILD_TAG = 0
_lock = threading.Lock()

# Buffer: lista de (texto, cor) onde cor é (R,G,B,A) ou None para branco
_LOG_BUFFER: list[tuple[str, tuple | None]] = []
# Cópia plain-text para o botão Copiar
_LOG_PLAIN: list[str] = []


# ---------------------------------------------------------------------------
# Classificação de cor por conteúdo da linha
# ---------------------------------------------------------------------------

def _classify(line: str) -> tuple | None:
    low = line.lower()
    if "error" in low or "traceback" in low or "exception" in low or "erro" in low:
        return (255, 80, 80, 255)       # vermelho
    if "warning" in low or "aviso" in low or "warn" in low:
        return (255, 200, 60, 255)      # amarelo
    if "ok" in low and "finished" in low:
        return (80, 220, 120, 255)      # verde
    if line.startswith("[runner]"):
        return (120, 180, 255, 255)     # azul claro
    if line.startswith("->") or line.startswith("<-"):
        return (160, 160, 160, 255)     # cinza
    return None                         # branco padrão


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_log_panel():
    global _LOG_CHILD_TAG
    _LOG_CHILD_TAG = dpg.generate_uuid()

    with dpg.group(horizontal=True):
        dpg.add_text(t("exec_log"))
        dpg.add_button(label=t("clear"), callback=_clear_log)
        dpg.add_button(label=t("copy"),  callback=_copy_log)

    dpg.add_separator()
    dpg.add_child_window(tag=_LOG_CHILD_TAG, width=-1, height=-1, border=False)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def append_log(line: str, color: tuple | None = "auto"):
    """Thread-safe: enfileira linha para o próximo frame."""
    if color == "auto":
        color = _classify(line)
    ts = datetime.now().strftime("%H:%M:%S")
    full = f"[{ts}] {line}"
    with _lock:
        _LOG_BUFFER.append((full, color))
        _LOG_PLAIN.append(full)


def flush_log_buffer():
    """Chamado no render loop da thread principal."""
    if not _LOG_BUFFER:
        return
    with _lock:
        entries = _LOG_BUFFER.copy()
        _LOG_BUFFER.clear()

    for text, color in entries:
        kwargs = {"default_value": text}
        if color:
            kwargs["color"] = color
        dpg.add_text(parent=_LOG_CHILD_TAG, **kwargs)

    # Auto-scroll
    if dpg.does_item_exist(_LOG_CHILD_TAG):
        dpg.set_y_scroll(_LOG_CHILD_TAG, dpg.get_y_scroll_max(_LOG_CHILD_TAG))


# ---------------------------------------------------------------------------
# Botões
# ---------------------------------------------------------------------------

def _clear_log():
    dpg.delete_item(_LOG_CHILD_TAG, children_only=True)
    with _lock:
        _LOG_PLAIN.clear()


def _copy_log():
    with _lock:
        text = "\n".join(_LOG_PLAIN)
    dpg.set_clipboard_text(text)
