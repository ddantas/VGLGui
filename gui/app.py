import os
import dearpygui.dearpygui as dpg
from dataclasses import dataclass, field
from gui.config import load_config
from gui.i18n import set_language, t

# ---------------------------------------------------------------------------
# Estado global da aplicação
# ---------------------------------------------------------------------------

@dataclass
class ProcedureState:
    name: str
    glyph_id: str              # ID do nó no canvas principal
    pos_x: float
    pos_y: float
    dpg_node_tag: int          # tag do nó no canvas principal
    dpg_attr_tags: dict        # {"i_in": tag, "o_out": tag}
    glyphs: dict = field(default_factory=dict)         # glyph_id → GlyphState
    links: dict = field(default_factory=dict)          # link_tag → (out_attr, in_attr)
    node_editor_tag: int = 0   # DPG node_editor dentro do popup
    window_tag: int = 0        # popup window tag (0 = não aberto)
    next_glyph_id: int = 1     # contador interno de IDs
    next_link_id: int = 0      # contador interno de links
    attr_tag_to_port: dict = field(default_factory=dict)  # tag → (glyph_id, port_key)
    ext_in_glyph_id: str = ""  # ID do glyph ExternalInput interno
    ext_out_glyph_id: str = "" # ID do glyph ExternalOutput interno


@dataclass
class GlyphState:
    glyph_id: str
    func: str
    library: str
    pos_x: float
    pos_y: float
    params: dict
    status: str = "ready"          # "ready" | "running" | "done" | "error"
    dpg_node_tag: int = 0
    dpg_attr_tags: dict = field(default_factory=dict)  # port_key → dpg tag
    preview_attr_tag: int = 0      # node_attribute contendo o thumbnail
    preview_tex_tag:  int = 0      # texture DPG do thumbnail


APP_STATE: dict = {
    "wksp_path":    None,    # str | None
    "dirty":        False,
    "device":       "GPU",

    "glyphs":       {},      # glyph_id → GlyphState
    "links":        {},      # link_tag → (out_attr_tag, in_attr_tag)

    "exec_process": None,
    "exec_running": False,

    "next_glyph_id": 1,

    "procedures":       {},    # proc_name → ProcedureState
    "active_procedure": None,  # None = canvas principal; str = editando procedure
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    # Load persisted config and apply language before building UI
    _cfg = load_config()
    set_language(_cfg.get("language", "pt"))

    from gui.canvas        import setup_canvas, flush_status_queue, bind_canvas_handlers, flush_node_previews
    from gui.sidebar_glyphs import setup_sidebar
    from gui.toolbar       import setup_menu_bar
    from gui.log_panel     import setup_log_panel, flush_log_buffer
    from gui.image_preview import setup_texture_registry, flush_preview_queue
    from gui.status_bar    import setup_status_bar, update_status_bar

    dpg.create_context()
    dpg.create_viewport(title="VGLGui", width=1400, height=900)
    dpg.setup_dearpygui()
    setup_texture_registry()

    with dpg.window(tag=1, no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True):
        setup_menu_bar()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=210, border=True, tag="sidebar_win"):
                setup_sidebar()

            with dpg.child_window(border=True, tag="canvas_win",
                                  height=-178,
                                  no_scrollbar=True,
                                  no_scroll_with_mouse=True):
                setup_canvas()

        with dpg.child_window(border=True, tag="log_win", height=128):
            setup_log_panel()

        with dpg.child_window(border=True, height=22, tag="status_bar_win", no_scrollbar=True):
            setup_status_bar()

    dpg.set_primary_window(1, True)
    bind_canvas_handlers()
    dpg.show_viewport()

    _last_title = ""
    while dpg.is_dearpygui_running():
        flush_log_buffer()
        flush_status_queue()
        flush_preview_queue()
        flush_node_previews()
        update_status_bar()

        # Atualiza título da janela com nome do arquivo e indicador de não-salvo
        path  = APP_STATE["wksp_path"]
        dirty = APP_STATE["dirty"]
        name  = os.path.basename(path) if path else t("untitled")
        title = f"{'* ' if dirty else ''}{name} — VGLGui"
        if title != _last_title:
            dpg.set_viewport_title(title)
            _last_title = title

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
