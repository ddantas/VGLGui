import dearpygui.dearpygui as dpg
from gui.glyph_registry import GLYPH_REGISTRY, CATEGORIES

# Tags dos botões de cada função: func → button_tag (para filtro de busca)
_func_button_tags: dict[str, int] = {}
_search_tag = 3001
_tag_counter = 3100


def _new_tag() -> int:
    global _tag_counter
    _tag_counter += 1
    return _tag_counter


def setup_sidebar():
    dpg.add_text("Funções VGL", tag=_new_tag())
    dpg.add_separator()
    dpg.add_input_text(
        tag=_search_tag,
        hint="Buscar...",
        width=-1,
        callback=_on_search,
    )
    dpg.add_separator()

    for category in CATEGORIES:
        funcs = [g for g in GLYPH_REGISTRY.values() if g.category == category]
        if not funcs:
            continue

        header_tag = _new_tag()
        with dpg.collapsing_header(label=category, tag=header_tag,
                                   default_open=(category in ("I/O", "Alocação", "Morfologia", "Filtros"))):
            for glyph_def in funcs:
                btn_tag = _new_tag()
                _func_button_tags[glyph_def.func] = btn_tag
                dpg.add_button(
                    label=glyph_def.label,
                    tag=btn_tag,
                    width=-1,
                    callback=_make_add_callback(glyph_def.func),
                )


def _make_add_callback(func: str):
    """Closure: cria callback que adiciona um Glyph ao canvas."""
    def _cb():
        from gui.canvas import add_glyph_to_canvas
        # Posiciona no centro aproximado do viewport visível
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
