import os
import sys
import dearpygui.dearpygui as dpg
from gui.app import APP_STATE
from gui.i18n import t

_RECENT_MENU_TAG: int = 0

# Raiz do projeto (um nível acima de gui/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Exemplos agrupados por categoria: (label, caminho relativo à raiz)
_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "2D": [
        ("Demo - Convolucao + Dilate + Erode",  "SAMPLES/demo.wksp"),
        ("Demo Gray",                            "SAMPLES/workflow_files/demoGray.wksp"),
        ("Demo RGB",                             "SAMPLES/workflow_files/demoRGB.wksp"),
        ("Drive (Segmentacao)",                  "SAMPLES/workflow_files/drive_.wksp"),
    ],
    "Fundus": [
        ("Fundus - Retina",                      "SAMPLES/fundus.wksp"),
        ("Fundus v2",                            "SAMPLES/workflow_files/fundus_v2.wksp"),
    ],
    "3D": [
        ("3D Demo",                              "SAMPLES/3d/3ddemo.wksp"),
        ("3D Functions",                         "SAMPLES/3d/functions3d.wksp"),
    ],
    "ND": [
        ("ND Basic",                             "SAMPLES/nd/nd.wksp"),
        ("ND Total",                             "SAMPLES/nd/ndtotal.wksp"),
        ("ND Shape + Strel (type)",              "SAMPLES/nd/nd_Strel_type.wksp"),
        ("ND Shape + Strel (window)",            "SAMPLES/nd/nd_Strel_window.wksp"),
    ],
    "Procedures": [
        ("GrayscaleConvert (procedure)",         "SAMPLES/procedures/exemplo_procedure_gui.wksp"),
        ("Demo Fundus (procedure)",              "SAMPLES/procedures/tcc/demo_fundus.wksp"),
    ],
}

def _new_tag() -> int:
    return dpg.generate_uuid()


def _populate_recent_menu():
    from gui.config import get_recent_files
    if not _RECENT_MENU_TAG or not dpg.does_item_exist(_RECENT_MENU_TAG):
        return
    dpg.delete_item(_RECENT_MENU_TAG, children_only=True)
    recent = get_recent_files()
    if not recent:
        dpg.add_text(t("recent_empty"), parent=_RECENT_MENU_TAG)
        return
    for path in recent:
        label = os.path.basename(path)
        dpg.add_menu_item(
            label=label,
            callback=_make_recent_callback(path),
            parent=_RECENT_MENU_TAG,
        )


def _make_recent_callback(path: str):
    def _cb():
        import os as _os
        if not _os.path.isfile(path):
            from gui.log_panel import append_log
            append_log(f"[recent] File not found: {path}")
            return
        from gui.wksp_io import load_wksp
        load_wksp(path)
    return _cb


def setup_menu_bar():
    with dpg.menu_bar():
        with dpg.menu(label=t("menu_file")):
            dpg.add_menu_item(label=t("new"),     callback=on_new)
            dpg.add_menu_item(label=t("open"),    callback=on_open)
            dpg.add_menu_item(label=t("save"),    callback=on_save)
            dpg.add_menu_item(label=t("save_as"),       callback=on_save_as)
            global _RECENT_MENU_TAG
            _RECENT_MENU_TAG = dpg.generate_uuid()
            with dpg.menu(label=t("recent_files"), tag=_RECENT_MENU_TAG):
                dpg.add_text(t("recent_empty"), tag=dpg.generate_uuid())
            _populate_recent_menu()
            dpg.add_menu_item(label=t("export_canvas"), callback=_export_canvas_image)
            dpg.add_separator()
            dpg.add_menu_item(label=t("exit"),    callback=dpg.stop_dearpygui)

        with dpg.menu(label=t("menu_run")):
            dpg.add_menu_item(label=t("run_gpu"),
                              callback=lambda: _run("GPU"))
            dpg.add_menu_item(label=t("run_cpu"),
                              callback=lambda: _run("CPU"))
            dpg.add_menu_item(label=t("stop"),
                              callback=_stop)

        with dpg.menu(label=t("menu_examples")):
            for category, examples in _EXAMPLES.items():
                with dpg.menu(label=category):
                    for label, rel_path in examples:
                        abs_path = os.path.join(_PROJECT_ROOT, rel_path)
                        if os.path.isfile(abs_path):
                            dpg.add_menu_item(
                                label=label,
                                callback=_make_example_callback(abs_path),
                            )
                        else:
                            dpg.add_menu_item(
                                label=f"{label}  {t('not_found')}",
                                enabled=False,
                            )

        with dpg.menu(label=t("menu_view")):
            dpg.add_menu_item(label=t("layout_lr"),
                              callback=lambda: _auto_layout("LR"))
            dpg.add_menu_item(label=t("layout_tb"),
                              callback=lambda: _auto_layout("TB"))
            dpg.add_menu_item(label=t("fit_screen"),
                              callback=_fit_screen)
            dpg.add_menu_item(label=t("toggle_log"),
                              callback=_toggle_log)
            dpg.add_menu_item(label=t("toggle_cat_colors"),
                              callback=_toggle_cat_colors)

        with dpg.menu(label=t("menu_language")):
            dpg.add_menu_item(label=t("lang_pt"), callback=lambda: _switch_language("pt"))
            dpg.add_menu_item(label=t("lang_en"), callback=lambda: _switch_language("en"))

        with dpg.menu(label=t("menu_help")):
            dpg.add_menu_item(label=t("help_shortcuts"), callback=_show_shortcuts)
            dpg.add_menu_item(label=t("help_about"),     callback=_show_about)

    _register_key_handlers()


def _ctrl():
    return dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

def _shift():
    return dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

def _register_key_handlers():
    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_N,
            callback=lambda s, d: on_new() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_O,
            callback=lambda s, d: on_open() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_S,
            callback=lambda s, d: on_save() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_F5,
            callback=lambda s, d: _run("GPU"))
        dpg.add_key_press_handler(dpg.mvKey_F6,
            callback=lambda s, d: _stop())
        dpg.add_key_press_handler(dpg.mvKey_L,
            callback=lambda s, d: _auto_layout() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_F,
            callback=lambda s, d: _fit_screen() if _ctrl() and _shift() else None)
        dpg.add_key_press_handler(dpg.mvKey_Z,
            callback=lambda s, d: _undo() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_Y,
            callback=lambda s, d: _redo() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_Delete,
            callback=lambda s, d: _delete_selected())
        dpg.add_key_press_handler(dpg.mvKey_C,
            callback=lambda s, d: _copy_nodes() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_V,
            callback=lambda s, d: _paste_nodes() if _ctrl() else None)
        dpg.add_key_press_handler(dpg.mvKey_D,
            callback=lambda s, d: _duplicate_nodes() if _ctrl() else None)


# ---------------------------------------------------------------------------
# Language switch
# ---------------------------------------------------------------------------

def _switch_language(lang: str):
    from gui.config import load_config, save_config
    cfg = load_config()
    if cfg.get("language") == lang:
        return
    cfg["language"] = lang
    save_config(cfg)
    # Restart the process to rebuild the UI with the new language
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# Exemplos
# ---------------------------------------------------------------------------

def _make_example_callback(abs_path: str):
    def _cb():
        if APP_STATE["dirty"] and APP_STATE["glyphs"]:
            _confirm_dialog(
                msg=t("unsaved_load"),
                on_yes=lambda: _load_example(abs_path),
            )
        else:
            _load_example(abs_path)
    return _cb


def _load_example(abs_path: str):
    import traceback
    from gui.wksp_io import load_wksp
    from gui.log_panel import append_log
    try:
        load_wksp(abs_path)
        from gui.config import add_recent_file
        add_recent_file(abs_path)
        _populate_recent_menu()
    except Exception as e:
        append_log(f"[toolbar] ERROR loading example:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Callbacks de File
# ---------------------------------------------------------------------------

def on_new():
    if APP_STATE["dirty"]:
        _confirm_dialog(
            msg=t("unsaved_new"),
            on_yes=_do_new,
        )
    else:
        _do_new()


def _do_new():
    from gui.wksp_io import new_workspace
    new_workspace()


def on_open():
    def _on_select(s, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            from gui.wksp_io import load_wksp
            load_wksp(path)
            from gui.config import add_recent_file
            add_recent_file(path)
            _populate_recent_menu()

    with dpg.file_dialog(
        label=t("open_workflow"),
        modal=True,
        width=600, height=400,
        callback=_on_select,
        tag=_new_tag(),
    ):
        dpg.add_file_extension(".wksp", color=(255, 200, 0, 255), custom_text="Workflow")
        dpg.add_file_extension(".*")


def on_save():
    path = APP_STATE.get("wksp_path")
    if path:
        from gui.wksp_io import save_wksp
        save_wksp(path)
    else:
        on_save_as()


def on_save_as():
    def _on_select(s, app_data):
        path = app_data.get("file_path_name", "")
        if path:
            if not path.endswith(".wksp"):
                path += ".wksp"
            from gui.wksp_io import save_wksp
            save_wksp(path)
            from gui.config import add_recent_file
            add_recent_file(path)
            _populate_recent_menu()

    with dpg.file_dialog(
        label=t("save_workflow"),
        modal=True,
        width=600, height=400,
        callback=_on_select,
        tag=_new_tag(),
        directory_selector=False,
    ):
        dpg.add_file_extension(".wksp", color=(255, 200, 0, 255), custom_text="Workflow")


# ---------------------------------------------------------------------------
# Export canvas as image
# ---------------------------------------------------------------------------

def _export_canvas_image():
    """Show a save-file dialog and capture the canvas_win region as a PNG."""

    def _on_select(s, app_data):
        path = app_data.get("file_path_name", "")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        _do_capture(path)

    with dpg.file_dialog(
        label=t("export_canvas_dialog"),
        modal=True,
        width=600, height=400,
        callback=_on_select,
        tag=_new_tag(),
        directory_selector=False,
    ):
        dpg.add_file_extension(".png", color=(100, 220, 100, 255), custom_text="PNG Image")


def _do_capture(path: str):
    """Capture the canvas_win region and save to *path* (PNG)."""
    import subprocess
    from gui.log_panel import append_log

    # Determine canvas region via DPG geometry queries
    try:
        pos  = dpg.get_item_rect_min("canvas_win")   # [x, y] in screen coords
        size = dpg.get_item_rect_size("canvas_win")  # [w, h]
        x, y, w, h = int(pos[0]), int(pos[1]), int(size[0]), int(size[1])
    except Exception:
        # Fallback: use pos + size separately
        try:
            pos  = dpg.get_item_pos("canvas_win")
            size = dpg.get_item_rect_size("canvas_win")
            x, y, w, h = int(pos[0]), int(pos[1]), int(size[0]), int(size[1])
        except Exception as e:
            append_log(t("export_canvas_error", err=str(e)))
            return

    if w <= 0 or h <= 0:
        append_log(t("export_canvas_error", err="canvas_win has zero size"))
        return

    # --- Method 1: mss ---
    try:
        import mss                              # type: ignore
        import mss.tools                        # type: ignore
        with mss.mss() as sct:
            monitor = {"top": y, "left": x, "width": w, "height": h}
            screenshot = sct.grab(monitor)
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=path)
        append_log(t("export_canvas_saved", path=path))
        return
    except ImportError:
        pass
    except Exception as e:
        append_log(t("export_canvas_error", err=f"mss: {e}"))
        return

    # --- Method 2: scrot ---
    try:
        result = subprocess.run(
            ["scrot", "-a", f"{x},{y},{w},{h}", path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            append_log(t("export_canvas_saved", path=path))
            return
        else:
            raise RuntimeError(result.stderr.strip())
    except FileNotFoundError:
        pass  # scrot not installed
    except Exception as e:
        append_log(t("export_canvas_error", err=f"scrot: {e}"))
        return

    # --- Method 3: ImageMagick import ---
    try:
        result = subprocess.run(
            [
                "import",
                "-window", "root",
                "-crop", f"{w}x{h}+{x}+{y}",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            append_log(t("export_canvas_saved", path=path))
            return
        else:
            raise RuntimeError(result.stderr.strip())
    except FileNotFoundError:
        pass  # ImageMagick not installed
    except Exception as e:
        append_log(t("export_canvas_error", err=f"ImageMagick: {e}"))
        return

    # --- No tool available ---
    append_log(t("export_canvas_no_tool"))


# ---------------------------------------------------------------------------
# Callbacks de Run
# ---------------------------------------------------------------------------

def _run(device: str):
    APP_STATE["device"] = device
    from gui.runner import run_workflow
    run_workflow(device)


def _stop():
    from gui.runner import stop_workflow
    stop_workflow()


# ---------------------------------------------------------------------------
# Callbacks de View
# ---------------------------------------------------------------------------

def _auto_layout(direction: str = "LR"):
    from gui.canvas import auto_layout
    auto_layout(direction)


def _fit_screen():
    from gui.canvas import fit_to_screen
    fit_to_screen()


def _undo():
    from gui.history import undo
    undo()


def _redo():
    from gui.history import redo
    redo()


def _show_shortcuts():
    tag = dpg.generate_uuid()
    with dpg.window(label=t("help_shortcuts"), modal=True, tag=tag,
                    width=380, no_resize=True, pos=[400, 200]):
        shortcuts = [
            ("Ctrl+N",         t("new")),
            ("Ctrl+O",         t("open")),
            ("Ctrl+S",         t("save")),
            ("Ctrl+Shift+S",   t("save_as")),
            ("Ctrl+Z",         t("undo")),
            ("Ctrl+Y",         t("redo")),
            ("Ctrl+L",         t("layout_lr")),
            ("Ctrl+Shift+F",   t("fit_screen")),
            ("F5",             t("run_gpu")),
            ("F6",             t("run_stop")),
            ("Delete",         t("delete_selected")),
            ("Ctrl+Scroll",    t("zoom_hint")),
        ]
        for key, desc in shortcuts:
            with dpg.group(horizontal=True):
                dpg.add_text(f"{key:<20}", color=(180, 220, 255, 255))
                dpg.add_text(desc)
        dpg.add_separator()
        dpg.add_button(label=t("close"), callback=lambda: dpg.delete_item(tag))


def _show_about():
    tag = dpg.generate_uuid()
    with dpg.window(label=t("help_about"), modal=True, tag=tag,
                    width=340, no_resize=True, pos=[450, 250]):
        dpg.add_text("VGLGui", color=(120, 200, 255, 255))
        dpg.add_text(t("about_desc"))
        dpg.add_separator()
        dpg.add_text(t("about_framework"))
        dpg.add_text(t("about_backend"))
        dpg.add_separator()
        dpg.add_button(label=t("close"), callback=lambda: dpg.delete_item(tag))


def _delete_selected():
    """Remove todos os nós selecionados no node editor."""
    from gui.canvas import NODE_EDITOR_TAG, remove_glyph
    from gui.app import APP_STATE
    selected = dpg.get_selected_nodes(NODE_EDITOR_TAG)
    if not selected:
        return
    # Mapeia dpg_node_tag → glyph_id
    tag_to_gid = {s.dpg_node_tag: gid for gid, s in APP_STATE["glyphs"].items()}
    for node_tag in selected:
        gid = tag_to_gid.get(node_tag)
        if gid:
            remove_glyph(gid)


def _toggle_cat_colors():
    from gui.canvas import toggle_category_colors
    toggle_category_colors()


def _toggle_log():
    if dpg.does_item_exist("log_win"):
        if dpg.is_item_shown("log_win"):
            dpg.hide_item("log_win")
        else:
            dpg.show_item("log_win")


def _copy_nodes():
    from gui.canvas import copy_selected
    copy_selected()


def _paste_nodes():
    from gui.canvas import paste_nodes
    paste_nodes()


def _duplicate_nodes():
    from gui.canvas import duplicate_selected
    duplicate_selected()


# ---------------------------------------------------------------------------
# Dialog de confirmação
# ---------------------------------------------------------------------------

def _confirm_dialog(msg: str, on_yes):
    popup_tag = _new_tag()
    with dpg.window(label=t("confirm"), modal=True, tag=popup_tag,
                    width=300, no_resize=True, pos=[500, 350]):
        dpg.add_text(msg, wrap=280)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label=t("yes"), width=100, callback=lambda: (
                dpg.delete_item(popup_tag), on_yes()
            ))
            dpg.add_button(label=t("no"), width=100,
                           callback=lambda: dpg.delete_item(popup_tag))
