import dearpygui.dearpygui as dpg
from gui.app import APP_STATE
from gui.i18n import t
import os

_tags: dict[str, int] = {}

def setup_status_bar():
    global _tags
    with dpg.group(horizontal=True):
        _tags["file"]   = dpg.add_text("", color=(180, 220, 255, 255))
        dpg.add_text(" | ")
        _tags["nodes"]  = dpg.add_text("")
        dpg.add_text(" | ")
        _tags["links"]  = dpg.add_text("")
        dpg.add_text(" | ")
        _tags["zoom"]   = dpg.add_text("")
        dpg.add_text(" | ")
        _tags["device"] = dpg.add_text("")

def update_status_bar():
    if not _tags:
        return
    path   = APP_STATE.get("wksp_path")
    name   = os.path.basename(path) if path else t("untitled")
    n_nodes = len(APP_STATE.get("glyphs", {}))
    n_links = len(APP_STATE.get("links", {}))
    from gui.canvas import get_zoom_level
    zoom   = get_zoom_level()
    device = APP_STATE.get("device", "GPU")

    for key, val in [
        ("file",   name),
        ("nodes",  f"{t('status_nodes')}: {n_nodes}"),
        ("links",  f"{t('status_links')}: {n_links}"),
        ("zoom",   f"{t('status_zoom')}: {zoom:.2f}x"),
        ("device", f"{device}"),
    ]:
        tag = _tags.get(key)
        if tag and dpg.does_item_exist(tag):
            dpg.set_value(tag, val)
