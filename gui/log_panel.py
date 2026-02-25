import dearpygui.dearpygui as dpg
import threading

_LOG_TEXT_TAG = 201
_LOG_CLEAR_TAG = 202
_LOG_BUFFER: list[str] = []
_lock = threading.Lock()


def setup_log_panel():
    with dpg.group(horizontal=True):
        dpg.add_text("Log de Execução")
        dpg.add_button(label="Limpar", tag=_LOG_CLEAR_TAG, callback=_clear_log)
        dpg.add_button(label="Copiar", callback=_copy_log)

    dpg.add_separator()
    dpg.add_input_text(
        tag=_LOG_TEXT_TAG,
        multiline=True,
        readonly=True,
        width=-1,
        height=-1,
        default_value="",
        tab_input=False,
    )


def append_log(line: str):
    """Thread-safe: enfileira linha para ser exibida no próximo frame."""
    with _lock:
        _LOG_BUFFER.append(line)


def flush_log_buffer():
    """Chamado no render loop da thread principal."""
    if not _LOG_BUFFER:
        return
    with _lock:
        lines = _LOG_BUFFER.copy()
        _LOG_BUFFER.clear()

    current = dpg.get_value(_LOG_TEXT_TAG) or ""
    new_text = current + "\n".join(lines) + "\n"
    dpg.set_value(_LOG_TEXT_TAG, new_text)

    # Auto-scroll para o final — mover cursor para o final do texto
    dpg.set_x_scroll(_LOG_TEXT_TAG, dpg.get_x_scroll_max(_LOG_TEXT_TAG))


def _clear_log():
    dpg.set_value(_LOG_TEXT_TAG, "")


def _copy_log():
    text = dpg.get_value(_LOG_TEXT_TAG) or ""
    dpg.set_clipboard_text(text)
