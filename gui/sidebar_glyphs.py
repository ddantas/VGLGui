import dearpygui.dearpygui as dpg
from gui.app import APP_STATE
from gui.glyph_registry import GLYPH_REGISTRY, CATEGORIES
from gui.i18n import t, cat_label

# Tags dos botões de cada função: func → button_tag (para filtro de busca)
_func_button_tags: dict[str, int] = {}
_search_tag: int = 0   # inicializado em setup_sidebar (após create_context)


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
