"""
Undo/Redo para o canvas principal.
Estratégia: snapshot do estado (glyphs + links) antes de cada ação.
Máximo de 50 entradas no histórico.
"""
from gui.app import APP_STATE

_undo_stack: list[dict] = []
_redo_stack: list[dict] = []
_MAX = 50


# ---------------------------------------------------------------------------
# Captura / restauração de estado
# ---------------------------------------------------------------------------

def _capture() -> dict:
    from gui.canvas import get_all_node_positions, get_attr_tag_to_port
    positions = get_all_node_positions()
    attr_map  = get_attr_tag_to_port()

    glyphs_snap = {}
    for gid, state in APP_STATE["glyphs"].items():
        x, y = positions.get(gid, (state.pos_x, state.pos_y))
        glyphs_snap[gid] = {
            "func":    state.func,
            "library": state.library,
            "pos_x":   x,
            "pos_y":   y,
            "params":  dict(state.params),
        }

    links_snap = []
    for _, (out_attr, in_attr) in APP_STATE["links"].items():
        out_info = attr_map.get(out_attr)
        in_info  = attr_map.get(in_attr)
        if out_info and in_info:
            out_gid, out_key = out_info
            in_gid,  in_key  = in_info
            links_snap.append((
                out_gid, out_key.removesuffix("_out"),
                in_gid,  in_key.removesuffix("_in"),
            ))

    return {
        "glyphs":        glyphs_snap,
        "links":         links_snap,
        "next_glyph_id": APP_STATE["next_glyph_id"],
    }


def _restore(snapshot: dict):
    from gui.canvas import clear_canvas, add_glyph_to_canvas, connect_ports
    clear_canvas()
    APP_STATE["next_glyph_id"] = snapshot.get("next_glyph_id", 1)

    for gid, data in snapshot["glyphs"].items():
        add_glyph_to_canvas(
            func=data["func"],
            pos_x=data["pos_x"],
            pos_y=data["pos_y"],
            forced_id=gid,
            params=data["params"],
        )

    for out_gid, out_port, in_gid, in_port in snapshot["links"]:
        connect_ports(out_gid, out_port, in_gid, in_port)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def push_undo():
    """Salva o estado atual antes de uma ação. Limpa o redo stack."""
    _undo_stack.append(_capture())
    if len(_undo_stack) > _MAX:
        _undo_stack.pop(0)
    _redo_stack.clear()


def undo():
    if not _undo_stack:
        return
    _redo_stack.append(_capture())
    _restore(_undo_stack.pop())


def redo():
    if not _redo_stack:
        return
    _undo_stack.append(_capture())
    _restore(_redo_stack.pop())


def clear_history():
    _undo_stack.clear()
    _redo_stack.clear()
