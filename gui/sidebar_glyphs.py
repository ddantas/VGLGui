import dearpygui.dearpygui as dpg
from gui.app import APP_STATE
from gui.glyph_registry import GLYPH_REGISTRY, CATEGORIES
from gui.i18n import t, cat_label

# Tags dos botões de cada função: func → button_tag (para filtro de busca)
_func_button_tags: dict[str, int] = {}
_search_tag: int = 0   # inicializado em setup_sidebar (após create_context)

# ---------------------------------------------------------------------------
# Drag-and-drop state
# ---------------------------------------------------------------------------
_drag_state: dict = {
    "active":     False,
    "func":       None,
    "label_tag":  0,     # tag da floating window (0 = não existe)
}


def setup_sidebar():
    global _search_tag
    _search_tag = dpg.generate_uuid()

    dpg.add_text(t("vgl_functions"), tag=dpg.generate_uuid())
    dpg.add_separator()

    # Botão "Nova Procedure" no topo da sidebar
    dpg.add_button(
        label=t("proc_new"),
        width=-1,
        callback=_on_new_procedure,
    )
    dpg.add_separator()

    dpg.add_input_text(
        tag=_search_tag,
        hint=t("search_hint"),
        width=-1,
        callback=_on_search,
    )
    dpg.add_separator()

    for category in CATEGORIES:
        # Procedures são criadas via botão "Nova Procedure" — não mostrar na lista
        if category == "Procedures":
            continue

        funcs = [g for g in GLYPH_REGISTRY.values() if g.category == category]
        if not funcs:
            continue

        header_tag = dpg.generate_uuid()
        with dpg.collapsing_header(label=cat_label(category), tag=header_tag,
                                   default_open=(category in ("I/O", "Allocation", "Morphology", "Filters"))):
            for glyph_def in funcs:
                btn_tag = dpg.generate_uuid()
                _func_button_tags[glyph_def.func] = btn_tag
                dpg.add_button(
                    label=glyph_def.label,
                    tag=btn_tag,
                    width=-1,
                    callback=_make_add_callback(glyph_def.func),
                    # mouse_down_callback used for drag detection via item handler below
                )


def _on_new_procedure():
    """Abre dialog para criar uma nova procedure."""
    from gui.procedure_canvas import show_new_procedure_dialog
    show_new_procedure_dialog()


def _make_add_callback(func: str):
    """Closure: cria callback que adiciona um Glyph ao canvas ou à procedure ativa."""
    def _cb():
        active_proc = APP_STATE.get("active_procedure")
        if active_proc:
            # Adiciona dentro da procedure que está sendo editada
            from gui.procedure_canvas import add_glyph_to_procedure
            add_glyph_to_procedure(active_proc, func, pos_x=300, pos_y=200)
        else:
            from gui.canvas import add_glyph_to_canvas
            vp_w = dpg.get_viewport_width()
            vp_h = dpg.get_viewport_height()
            add_glyph_to_canvas(func, pos_x=vp_w // 2 - 100, pos_y=vp_h // 2 - 60)
    return _cb


def _on_search(sender, value: str):
    """Filtra botões da sidebar conforme o texto digitado."""
    query = value.strip().lower()
    for func, btn_tag in _func_button_tags.items():
        if not dpg.does_item_exist(btn_tag):
            continue
        if query == "" or query in func.lower():
            dpg.show_item(btn_tag)
        else:
            dpg.hide_item(btn_tag)


# ---------------------------------------------------------------------------
# Drag-and-drop helpers
# ---------------------------------------------------------------------------

def _make_drag_start_callback(func: str):
    """Retorna callback de mouse_down que inicia o drag de um glyph."""
    def _cb(sender, app_data):
        # app_data for mouse_down = mouse button index (0 = left)
        if app_data != 0:
            return
        _start_drag(func)
    return _cb


def _start_drag(func: str):
    """Inicia o estado de drag e cria a floating label."""
    global _drag_state
    if _drag_state["active"]:
        return  # já está em drag

    glyph_def = GLYPH_REGISTRY.get(func)
    label_text = glyph_def.label if glyph_def else func

    mx, my = dpg.get_mouse_pos(local=False)

    win_tag = dpg.generate_uuid()
    dpg.add_window(
        tag=win_tag,
        pos=[int(mx) + 12, int(my) + 12],
        width=160,
        height=28,
        no_title_bar=True,
        no_resize=True,
        no_move=True,
        no_scrollbar=True,
        no_close=True,
        no_background=False,
    )
    dpg.add_text(f"+ {label_text}", parent=win_tag)

    _drag_state["active"]    = True
    _drag_state["func"]      = func
    _drag_state["label_tag"] = win_tag


def _cancel_drag():
    """Cancela o drag e destrói a floating label."""
    global _drag_state
    if _drag_state["label_tag"] and dpg.does_item_exist(_drag_state["label_tag"]):
        dpg.delete_item(_drag_state["label_tag"])
    _drag_state["active"]    = False
    _drag_state["func"]      = None
    _drag_state["label_tag"] = 0


def _on_drag_move(sender, app_data):
    """Atualiza posição da floating label enquanto o drag está ativo."""
    if not _drag_state["active"]:
        return
    win_tag = _drag_state["label_tag"]
    if not win_tag or not dpg.does_item_exist(win_tag):
        return
    mx, my = dpg.get_mouse_pos(local=False)
    dpg.set_item_pos(win_tag, [int(mx) + 12, int(my) + 12])


def _on_drag_release(sender, app_data):
    """Finaliza o drag: se o mouse estiver sobre canvas_win, adiciona o glyph."""
    if not _drag_state["active"]:
        return

    func = _drag_state["func"]
    _cancel_drag()

    if func is None:
        return

    # Verifica se o mouse está sobre o canvas_win
    if not dpg.does_item_exist("canvas_win"):
        return
    if not dpg.is_item_hovered("canvas_win"):
        return

    # Calcula posição relativa ao canvas_win
    mx, my = dpg.get_mouse_pos(local=False)
    try:
        rect_min = dpg.get_item_rect_min("canvas_win")
        cx = mx - rect_min[0]
        cy = my - rect_min[1]
    except Exception:
        try:
            pos = dpg.get_item_pos("canvas_win")
            cx = mx - pos[0]
            cy = my - pos[1]
        except Exception:
            cx, cy = mx, my

    active_proc = APP_STATE.get("active_procedure")
    if active_proc:
        from gui.procedure_canvas import add_glyph_to_procedure
        add_glyph_to_procedure(active_proc, func, pos_x=cx, pos_y=cy)
    else:
        from gui.canvas import add_glyph_to_canvas
        add_glyph_to_canvas(func, pos_x=cx, pos_y=cy)


def bind_sidebar_handlers():
    """Registra handlers globais de mouse para drag-and-drop da sidebar.

    Deve ser chamado de app.py após bind_canvas_handlers(), fora de qualquer
    contexto de janela/widget.
    """
    # Registra mouse_down em cada botão de função via item_handler_registry
    for func, btn_tag in _func_button_tags.items():
        if not dpg.does_item_exist(btn_tag):
            continue
        registry_tag = dpg.generate_uuid()
        with dpg.item_handler_registry(tag=registry_tag):
            dpg.add_item_clicked_handler(
                button=0,
                callback=_make_drag_start_callback(func),
            )
        dpg.bind_item_handler_registry(btn_tag, registry_tag)

    # Handlers globais de mouse para mover e soltar
    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=_on_drag_move)
        dpg.add_mouse_release_handler(callback=_on_drag_release)
