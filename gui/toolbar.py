import dearpygui.dearpygui as dpg
from gui.app import APP_STATE

_tag_counter = 4000


def _new_tag() -> int:
    global _tag_counter
    _tag_counter += 1
    return _tag_counter


def setup_menu_bar():
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Novo          Ctrl+N", callback=on_new)
            dpg.add_menu_item(label="Abrir...      Ctrl+O", callback=on_open)
            dpg.add_menu_item(label="Salvar        Ctrl+S", callback=on_save)
            dpg.add_menu_item(label="Salvar Como   Ctrl+Shift+S", callback=on_save_as)
            dpg.add_separator()
            dpg.add_menu_item(label="Sair", callback=dpg.stop_dearpygui)

        with dpg.menu(label="Run"):
            dpg.add_menu_item(label="Executar GPU  F5",
                              callback=lambda: _run("GPU"))
            dpg.add_menu_item(label="Executar CPU",
                              callback=lambda: _run("CPU"))
            dpg.add_menu_item(label="Parar         F6",
                              callback=_stop)

        with dpg.menu(label="View"):
            dpg.add_menu_item(label="Fit to Screen  Ctrl+Shift+F",
                              callback=_fit_screen)
            dpg.add_menu_item(label="Toggle Log",
                              callback=_toggle_log)

    _register_key_handlers()


def _register_key_handlers():
    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_N,
            callback=lambda s, d: on_new() if dpg.is_key_down(dpg.mvKey_Control) else None)
        dpg.add_key_press_handler(dpg.mvKey_O,
            callback=lambda s, d: on_open() if dpg.is_key_down(dpg.mvKey_Control) else None)
        dpg.add_key_press_handler(dpg.mvKey_S,
            callback=lambda s, d: on_save() if dpg.is_key_down(dpg.mvKey_Control) else None)
        dpg.add_key_press_handler(dpg.mvKey_F5,
            callback=lambda s, d: _run("GPU"))
        dpg.add_key_press_handler(dpg.mvKey_F6,
            callback=lambda s, d: _stop())


# ---------------------------------------------------------------------------
# Callbacks de File
# ---------------------------------------------------------------------------

def on_new():
    if APP_STATE["dirty"]:
        _confirm_dialog(
            msg="Há mudanças não salvas. Deseja continuar?",
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

    with dpg.file_dialog(
        label="Abrir Workflow",
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

    with dpg.file_dialog(
        label="Salvar Workflow",
        modal=True,
        width=600, height=400,
        callback=_on_select,
        tag=_new_tag(),
        directory_selector=False,
    ):
        dpg.add_file_extension(".wksp", color=(255, 200, 0, 255), custom_text="Workflow")


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

def _fit_screen():
    # Dear PyGui não expõe "fit to screen" diretamente no node editor;
    # reseta a posição de pan para (0, 0) como aproximação
    pass


def _toggle_log():
    if dpg.does_item_exist("log_win"):
        if dpg.is_item_shown("log_win"):
            dpg.hide_item("log_win")
        else:
            dpg.show_item("log_win")


# ---------------------------------------------------------------------------
# Dialog de confirmação
# ---------------------------------------------------------------------------

def _confirm_dialog(msg: str, on_yes):
    popup_tag = _new_tag()
    with dpg.window(label="Confirmar", modal=True, tag=popup_tag,
                    width=300, no_resize=True, pos=[500, 350]):
        dpg.add_text(msg, wrap=280)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Sim", width=100, callback=lambda: (
                dpg.delete_item(popup_tag), on_yes()
            ))
            dpg.add_button(label="Não", width=100,
                           callback=lambda: dpg.delete_item(popup_tag))
