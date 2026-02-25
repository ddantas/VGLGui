import dearpygui.dearpygui as dpg
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Estado global da aplicação
# ---------------------------------------------------------------------------

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


APP_STATE: dict = {
    "wksp_path":    None,    # str | None
    "dirty":        False,
    "device":       "GPU",

    "glyphs":       {},      # glyph_id → GlyphState
    "links":        {},      # link_tag → (out_attr_tag, in_attr_tag)

    "exec_process": None,
    "exec_running": False,

    "next_glyph_id": 1,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    from gui.canvas       import setup_canvas
    from gui.sidebar_glyphs import setup_sidebar
    from gui.toolbar      import setup_menu_bar
    from gui.log_panel    import setup_log_panel, flush_log_buffer

    dpg.create_context()
    dpg.create_viewport(title="VGLGui", width=1400, height=900)
    dpg.setup_dearpygui()

    with dpg.window(tag=1, no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True):
        setup_menu_bar()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=210, border=True, tag="sidebar_win"):
                setup_sidebar()

            with dpg.child_window(border=True, tag="canvas_win",
                                  height=-160):
                setup_canvas()

        with dpg.child_window(border=True, tag="log_win", height=150):
            setup_log_panel()

    dpg.set_primary_window(1, True)
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        flush_log_buffer()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
