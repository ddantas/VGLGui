import os
import sys
import dearpygui.dearpygui as dpg
from gui.app import APP_STATE
from gui.glyph_registry import GLYPH_REGISTRY
from gui.i18n import t


def _topological_sort(glyphs: dict, links: dict, attr_map: dict) -> list:
    """Retorna glyph_ids em ordem topológica (fontes primeiro)."""
    from collections import deque

    glyph_ids = list(glyphs.keys())
    in_degree  = {gid: 0 for gid in glyph_ids}
    successors = {gid: [] for gid in glyph_ids}

    for out_attr, in_attr in links.values():
        out_info = attr_map.get(out_attr)
        in_info  = attr_map.get(in_attr)
        if not out_info or not in_info:
            continue
        out_gid, in_gid = out_info[0], in_info[0]
        if out_gid != in_gid and out_gid in successors and in_gid in successors:
            successors[out_gid].append(in_gid)
            in_degree[in_gid] += 1

    queue  = deque(gid for gid in glyph_ids if in_degree[gid] == 0)
    result = []
    while queue:
        gid = queue.popleft()
        result.append(gid)
        for succ in successors[gid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # Glyphs em ciclo ou isolados vão ao final
    remaining = [gid for gid in glyph_ids if gid not in result]
    return result + remaining


def save_wksp(path: str):
    """Serializa o estado atual do canvas para o formato .wksp."""
    from gui.canvas import get_all_node_positions, get_attr_tag_to_port
    from gui.procedure_canvas import get_all_procedure_positions

    positions = get_all_node_positions()
    attr_map  = get_attr_tag_to_port()  # tag → (glyph_id, port_key)

    lines = []
    lines.append("WorkspaceBegin: 1.0\n")
    lines.append("\nVariablesBegin:\n\nVariablesEnd:\n\n")

    # ── Glyphs normais do workspace principal (ordem topológica) ────────────
    sorted_ids = _topological_sort(APP_STATE["glyphs"], APP_STATE["links"], attr_map)
    for glyph_id in sorted_ids:
        state = APP_STATE["glyphs"][glyph_id]
        x, y = positions.get(glyph_id, (state.pos_x, state.pos_y))
        params_str = _build_params_str(state.params)
        lines.append(
            f"Glyph:{state.library}:{state.func}::localhost:"
            f"{glyph_id}:{int(x)}:{int(y)}::{params_str}\n"
        )

    # ── Blocos ProcedureBegin … ProcedureEnd ─────────────────────────────────
    for proc_name, proc in APP_STATE["procedures"].items():
        proc_positions = get_all_procedure_positions(proc_name)

        # Posição do nó procedure no canvas principal
        if dpg.does_item_exist(proc.dpg_node_tag):
            pos = dpg.get_item_pos(proc.dpg_node_tag)
            px, py = pos[0], pos[1]
        else:
            px, py = proc.pos_x, proc.pos_y

        lines.append(
            f"ProcedureBegin:{proc_name}:ProcedureBegin::localhost:"
            f"{proc.glyph_id}:{int(px)}:{int(py)}::\n"
        )

        # Glyphs internos
        for gid, state in proc.glyphs.items():
            ix, iy = proc_positions.get(gid, (state.pos_x, state.pos_y))
            if state.library == "ExtPort":
                direction = "out"
                lines.append(
                    f"ExtPort:{direction}:{state.func}::localhost:"
                    f"{gid}:{int(ix)}:{int(iy)}::\n"
                )
            else:
                params_str = _build_params_str(state.params)
                lines.append(
                    f"Glyph:{state.library}:{state.func}::localhost:"
                    f"{gid}:{int(ix)}:{int(iy)}::{params_str}\n"
                )

        # Conexões internas — usando dados estáveis (glyph_id, port_key)
        for link_id, (out_gid, out_key, in_gid, in_key) in proc.links.items():
            out_port = out_key.removesuffix("_out")
            in_port  = in_key.removesuffix("_in")
            lines.append(
                f"NodeConnection:data:{out_gid}:{out_port}:{in_gid}:{in_port}\n"
            )

        lines.append(f"ProcedureEnd:{proc_name}\n\n")

    lines.append("\n")

    # ── Conexões do workspace principal (inclui portas de procedures) ─────────
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
    print(f"[wksp_io] Saved to: {path}")


def _param_val_to_str(v) -> str:
    """Converte valor de parâmetro para string adequada ao formato .wksp."""
    if isinstance(v, list):
        return '[' + ','.join(str(x) for x in v) + ']'
    return str(v) if v is not None else ""


def _is_plain_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _build_params_str(params: dict) -> str:
    parts = []
    for name, value in params.items():
        if value == "" or value is None:
            continue
        val_str = str(value)
        if val_str == "True":
            val_str = "1"
        elif val_str == "False":
            val_str = "0"

        if val_str.startswith("["):
            parts.append(f"-{name} {val_str}")
        elif _is_plain_number(val_str):
            parts.append(f"-{name} {val_str}")
        else:
            parts.append(f"-{name} '{val_str}'")
    return " ".join(parts)


def load_wksp(path: str):
    """Carrega um arquivo .wksp e reconstrói o canvas."""
    from gui.canvas import clear_canvas, add_glyph_to_canvas, connect_ports
    from gui.procedure_canvas import (add_procedure_to_canvas,
                                      add_glyph_to_procedure,
                                      connect_ports_in_procedure,
                                      open_procedure_popup,
                                      _close_procedure_popup)
    from gui.log_panel import append_log

    if not os.path.isfile(path):
        append_log(t("wksp_not_found", path=path))
        return

    _orig_argv = sys.argv[:]
    sys.argv = ["gui", path]

    try:
        if "readWorkflow" in sys.modules:
            del sys.modules["readWorkflow"]

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from readWorkflow import Workspace, fileRead

        workspace = Workspace()
        fileRead(workspace)

    except Exception as e:
        append_log(t("wksp_error_read", path=path, err=e))
        sys.argv = _orig_argv
        return
    finally:
        sys.argv = _orig_argv

    clear_canvas()

    # ── 1. Criar nós de procedure para glyphs ProcedureBegin ─────────────────
    for glyph in workspace.lstGlyph:
        if glyph.func == "ProcedureBegin":
            add_procedure_to_canvas(
                name=glyph.library,          # nome está no campo library
                pos_x=float(glyph.glyph_x),
                pos_y=float(glyph.glyph_y),
                proc_glyph_id=str(glyph.glyph_id),
            )

    # ── 2. Reconstruir conteúdo interno de cada sub_workspace ─────────────────
    # IDs dos glyphs no workspace principal (para detectar referências legadas)
    main_glyph_ids = {str(g.glyph_id) for g in workspace.lstGlyph}
    # Mapeamentos legados: (ext_input_id_main, proc_glyph_id) para criar conexões depois
    legacy_proc_connections: list[tuple[str, str]] = []

    for sub_ws in getattr(workspace, "subWorkspaces", []):
        proc_name = getattr(sub_ws, "name", None)
        if not proc_name or proc_name not in APP_STATE["procedures"]:
            continue

        proc = APP_STATE["procedures"][proc_name]

        # IDs dos glyphs internos desta procedure (do arquivo)
        sub_glyph_ids = {str(g.glyph_id) for g in sub_ws.lstGlyph}

        # Encontra IDs dos ExtPorts no arquivo para sobrescrever os auto-alocados
        for g in sub_ws.lstGlyph:
            if g.func == "External Input (1)":
                proc.ext_in_glyph_id = str(g.glyph_id)
                # Atualiza next_glyph_id global para evitar colisão
                try:
                    num = int(g.glyph_id)
                    if num >= APP_STATE["next_glyph_id"]:
                        APP_STATE["next_glyph_id"] = num + 1
                except ValueError:
                    pass
            elif g.func == "External Output (1)":
                proc.ext_out_glyph_id = str(g.glyph_id)
                try:
                    num = int(g.glyph_id)
                    if num >= APP_STATE["next_glyph_id"]:
                        APP_STATE["next_glyph_id"] = num + 1
                except ValueError:
                    pass

        # Abre popup para criar node_editor e nós External I/O com IDs corretos
        open_procedure_popup(proc_name)

        # Adiciona glyphs internos (não ExtPort)
        for g in sub_ws.lstGlyph:
            if g.func in ("External Input (1)", "External Output (1)"):
                continue
            glyph_def = GLYPH_REGISTRY.get(g.func)
            if glyph_def and glyph_def.params:
                params = {}
                for i, p in enumerate(g.lst_par):
                    key = glyph_def.params[i].name if i < len(glyph_def.params) else p.getName()
                    params[key] = _param_val_to_str(p.getValue())
            else:
                params = {p.getName(): _param_val_to_str(p.getValue()) for p in g.lst_par}
            add_glyph_to_procedure(
                proc_name,
                g.func,
                pos_x=float(g.glyph_x),
                pos_y=float(g.glyph_y),
                forced_id=str(g.glyph_id),
                params=params,
            )

        # Reconecta portas internas, com suporte a formato legado
        # (ExtInput no workspace principal referenciado dentro da procedure)
        legacy_ext_in_id: str | None = None
        for conn in sub_ws.lstConnections:
            out_gid  = str(conn.output_glyph_id)
            out_port = conn.output_varname.strip()

            # Formato legado: glyph de saída pertence ao workspace principal
            actual_out_gid  = out_gid
            actual_out_port = out_port
            if out_gid not in sub_glyph_ids and out_gid in main_glyph_ids:
                # Remapeia para o ExtInput interno da procedure
                legacy_ext_in_id = out_gid
                actual_out_gid  = proc.ext_in_glyph_id
                actual_out_port = 'o'

            for inp in conn.lst_con_input:
                in_gid  = str(inp.Par_glyph_id)
                in_port = inp.Par_name.strip()
                # Pula conexões cujo destino é um glyph do workspace principal
                if in_gid not in sub_glyph_ids and in_gid in main_glyph_ids:
                    continue
                connect_ports_in_procedure(proc_name, actual_out_gid, actual_out_port, in_gid, in_port)

        # Registra para criar conexão legacy ExtInput→ProcedureBegin depois
        if legacy_ext_in_id is not None:
            legacy_proc_connections.append((legacy_ext_in_id, proc.glyph_id))

        # Fecha popup (usuário pode reabrir clicando "Abrir" no nó)
        _close_procedure_popup(proc_name)

    # ── 3. Reconstrói glyphs normais do workspace principal ──────────────────
    glyphs_ok = 0
    for glyph in workspace.lstGlyph:
        if glyph.func == "ProcedureBegin":
            glyphs_ok += 1   # conta o nó procedure
            continue
        glyph_def = GLYPH_REGISTRY.get(glyph.func)
        if glyph_def and glyph_def.params:
            params = {}
            for i, p in enumerate(glyph.lst_par):
                key = glyph_def.params[i].name if i < len(glyph_def.params) else p.getName()
                params[key] = _param_val_to_str(p.getValue())
        else:
            params = {p.getName(): _param_val_to_str(p.getValue()) for p in glyph.lst_par}
        try:
            result = add_glyph_to_canvas(
                func=glyph.func,
                pos_x=float(glyph.glyph_x),
                pos_y=float(glyph.glyph_y),
                forced_id=glyph.glyph_id,
                params=params,
            )
            if result:
                glyphs_ok += 1
            else:
                append_log(t("wksp_unknown_func", func=glyph.func))
        except Exception as e:
            append_log(t("wksp_error_glyph", func=glyph.func, err=e))

    # ── 3b. Conexões legadas: ExtInput do workspace principal → ProcedureBegin ─
    # Garante que procedures de formato legado recebam imagem via porta 'i'
    for ext_in_id, proc_glyph_id in legacy_proc_connections:
        connect_ports(ext_in_id, 'o', proc_glyph_id, 'i')

    # ── 4. Conexões do workspace principal (inclui portas de procedures) ──────
    links_ok = 0
    for conn in workspace.lstConnections:
        out_gid  = conn.output_glyph_id
        out_port = conn.output_varname.strip()
        for inp in conn.lst_con_input:
            in_gid  = inp.Par_glyph_id
            in_port = inp.Par_name.strip()
            ok = connect_ports(out_gid, out_port, in_gid, in_port)
            if ok:
                links_ok += 1

    APP_STATE["wksp_path"] = path
    APP_STATE["dirty"] = False
    total_glyphs = len(workspace.lstGlyph)
    append_log(t("wksp_loaded",
                 name=os.path.basename(path),
                 ok=glyphs_ok,
                 total=total_glyphs,
                 links=links_ok))

    # Auto-layout se houver posições sobrepostas (arquivos legados)
    if _has_overlapping_positions(workspace):
        try:
            from gui.canvas import auto_layout
            auto_layout()
        except Exception as e:
            import traceback
            append_log(t("wksp_layout_error", tb=traceback.format_exc()))


def _has_overlapping_positions(workspace) -> bool:
    """Retorna True se algum par de glyphs compartilha a mesma posição (x, y)."""
    seen = set()
    for g in workspace.lstGlyph:
        pos = (int(float(g.glyph_x)), int(float(g.glyph_y)))
        if pos in seen:
            return True
        seen.add(pos)
    return False


def new_workspace():
    from gui.canvas import clear_canvas
    clear_canvas()
    APP_STATE["wksp_path"] = None
    APP_STATE["dirty"] = False
