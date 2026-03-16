import os
import threading
import dearpygui.dearpygui as dpg
import numpy as np
from gui.i18n import t

_preview_queue: list[str] = []
_preview_lock  = threading.Lock()
_TEXTURE_REGISTRY_TAG = 0   # definido em setup_texture_registry()


def setup_texture_registry():
    global _TEXTURE_REGISTRY_TAG
    _TEXTURE_REGISTRY_TAG = dpg.generate_uuid()
    dpg.add_texture_registry(tag=_TEXTURE_REGISTRY_TAG)


def get_texture_registry_tag() -> int:
    return _TEXTURE_REGISTRY_TAG


def queue_show_image(path: str):
    """Thread-safe: enfileira caminho de imagem para exibir no próximo frame."""
    with _preview_lock:
        _preview_queue.append(path)


def flush_preview_queue():
    """Chamado no render loop da thread principal."""
    if not _preview_queue:
        return
    with _preview_lock:
        paths = _preview_queue.copy()
        _preview_queue.clear()
    for path in paths:
        try:
            _show_image_popup(path)
        except Exception as e:
            print(f"[image_preview] Error displaying {path}: {e}")


def _show_image_popup(path: str):
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    width, height = img.size
    data = np.array(img, dtype=np.float32) / 255.0

    # Redimensiona para caber na tela (máx 700px no lado maior)
    max_dim = 700
    scale   = min(max_dim / max(width, height), 1.0)
    disp_w  = max(int(width  * scale), 1)
    disp_h  = max(int(height * scale), 1)

    texture_tag = dpg.generate_uuid()
    dpg.add_static_texture(
        width=width, height=height,
        default_value=data.flatten().tolist(),
        tag=texture_tag,
        parent=_TEXTURE_REGISTRY_TAG,
    )

    popup_tag = dpg.generate_uuid()
    title = os.path.basename(path)

    def _close():
        if dpg.does_item_exist(popup_tag):
            dpg.delete_item(popup_tag)
        if dpg.does_item_exist(texture_tag):
            dpg.delete_item(texture_tag)
        try:
            os.unlink(path)
        except OSError:
            pass

    with dpg.window(
        label=title, tag=popup_tag,
        width=disp_w + 24, height=disp_h + 60,
        on_close=_close,
    ):
        dpg.add_image(texture_tag, width=disp_w, height=disp_h)
        dpg.add_button(label=t("close"), callback=_close)
