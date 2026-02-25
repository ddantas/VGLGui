import os
import sys
import dearpygui.dearpygui as dpg
from gui.app import APP_STATE


def save_wksp(path: str):
    """Serializa o estado atual do canvas para o formato .wksp."""
    from gui.canvas import get_all_node_positions, get_attr_tag_to_port

    positions = get_all_node_positions()
    attr_map  = get_attr_tag_to_port()  # tag → (glyph_id, port_key)

    lines = []
    lines.append("WorkspaceBegin: 1.0\n")
    lines.append("\nVariablesBegin:\n\nVariablesEnd:\n\n")

    for glyph_id, state in APP_STATE["glyphs"].items():
        x, y = positions.get(glyph_id, (state.pos_x, state.pos_y))

        # Monta string de parâmetros
        params_str = _build_params_str(state.params)

        lines.append(
            f"Glyph:{state.library}:{state.func}::localhost:"
            f"{glyph_id}:{int(x)}:{int(y)}::{params_str}\n"
        )

    lines.append("\n")

    # Inverte o mapeamento: (glyph_id, port_key) → tag
    port_to_tag: dict[tuple, int] = {v: k for k, v in attr_map.items()}

    for link_tag, (out_attr, in_attr) in APP_STATE["links"].items():
        out_info = attr_map.get(out_attr)  # (glyph_id, "portname_out")
        in_info  = attr_map.get(in_attr)   # (glyph_id, "portname_in")
        if not out_info or not in_info:
            continue
        out_gid, out_key = out_info
        in_gid,  in_key  = in_info
        out_port = out_key.removesuffix("_out")
        in_port  = in_key.removesuffix("_in")
        lines.append(
            f"NodeConnection:data:{out_gid}:{out_port}:{in_gid}:{in_port}\n"
        )

    lines.append("\nWorkspaceEnd: 1.0\n")

    with open(path, "w") as f:
        f.writelines(lines)

    APP_STATE["wksp_path"] = path
    APP_STATE["dirty"] = False
    print(f"[wksp_io] Salvo em: {path}")


def _build_params_str(params: dict) -> str:
    parts = []
    for name, value in params.items():
        if value == "" or value is None:
            continue
        val_str = str(value)
        # Valores com espaço ou que são arrays ficam entre aspas simples
        if " " in val_str or val_str.startswith("["):
            parts.append(f"-{name} '{val_str}'")
        else:
            parts.append(f"-{name} {val_str}")
    return " ".join(parts)


def load_wksp(path: str):
    """Carrega um arquivo .wksp e reconstrói o canvas."""
    from gui.canvas import clear_canvas, add_glyph_to_canvas, connect_ports

    if not os.path.isfile(path):
        print(f"[wksp_io] Arquivo não encontrado: {path}")
        return

    # readWorkflow usa sys.argv[1] como caminho do arquivo
    _orig_argv = sys.argv[:]
    sys.argv = ["gui", path]

    try:
        # Importa fresh — readWorkflow é um módulo com estado global
        if "readWorkflow" in sys.modules:
            del sys.modules["readWorkflow"]

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from readWorkflow import Workspace, fileRead

        workspace = Workspace()
        fileRead(workspace)

    except Exception as e:
        print(f"[wksp_io] Erro ao ler {path}: {e}")
        sys.argv = _orig_argv
        return
    finally:
        sys.argv = _orig_argv

    clear_canvas()

    # Reconstrói Glyphs
    for glyph in workspace.lstGlyph:
        params = {p.getName(): p.getValue() for p in glyph.lst_par}
        try:
            add_glyph_to_canvas(
                func=glyph.func,
                pos_x=float(glyph.glyph_x),
                pos_y=float(glyph.glyph_y),
                forced_id=glyph.glyph_id,
                params=params,
            )
        except Exception as e:
            print(f"[wksp_io] Não foi possível criar glyph '{glyph.func}': {e}")

    # Reconstrói conexões
    for conn in workspace.lstConnections:
        out_gid  = conn.output_glyph_id
        out_port = conn.output_varname.strip()
        for inp in conn.lst_con_input:
            in_gid  = inp.Par_glyph_id
            in_port = inp.Par_name.strip()
            connect_ports(out_gid, out_port, in_gid, in_port)

    APP_STATE["wksp_path"] = path
    APP_STATE["dirty"] = False
    print(f"[wksp_io] Carregado: {path}  "
          f"({len(workspace.lstGlyph)} glyphs, {len(workspace.lstConnections)} conexões)")


def new_workspace():
    from gui.canvas import clear_canvas
    clear_canvas()
    APP_STATE["wksp_path"] = None
    APP_STATE["dirty"] = False
