"""
procedure_canvas.py — gerencia procedures na GUI VGLGui.

Cada procedure aparece como um único nó no canvas principal (tema azul).
Clicar em "Abrir" abre um popup com o node_editor interno da procedure.
"""

import dearpygui.dearpygui as dpg
from gui.app import APP_STATE, ProcedureState, GlyphState
from gui.glyph_registry import GLYPH_REGISTRY, ParamDef
from gui.i18n import t

# Tema azul escuro para o nó procedure no canvas principal
_PROC_THEME_TAG = 0

# Mapeamento DPG link_tag → (proc_name, link_id) para remoção de links
_dpg_link_to_id: dict[int, tuple[str, int]] = {}


def _new_tag() -> int:
    return dpg.generate_uuid()


def init_procedure_theme():
    """Cria o tema visual para nós de procedure. Chamar após create_context."""
    global _PROC_THEME_TAG
    _PROC_THEME_TAG = _new_tag()
    with dpg.theme(tag=_PROC_THEME_TAG):
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,
                                (20, 60, 120, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,
                                (30, 80, 150, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected,
                                (40, 100, 180, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,
                                (10, 40, 100, 255),
                                category=dpg.mvThemeCat_Nodes)


# ---------------------------------------------------------------------------
# Nó de procedure no canvas principal
# ---------------------------------------------------------------------------

def add_procedure_to_canvas(name: str, pos_x: float = 200, pos_y: float = 200,
                             proc_glyph_id: str = None) -> str | None:
    """
    Cria o nó ProcedureBegin no canvas principal.
    Retorna o glyph_id (string) ou None em caso de erro.
    """
    from gui.canvas import NODE_EDITOR_TAG
    from gui.log_panel import append_log

    if not name:
        return None

    # Garante que o nome é único
    if name in APP_STATE["procedures"]:
        append_log(f"[procedure] Já existe uma procedure chamada '{name}'")
        return None

    # Gera ou usa o ID fornecido
    if proc_glyph_id is not None:
        glyph_id = str(proc_glyph_id)
        try:
            numeric = int(glyph_id)
            if numeric >= APP_STATE["next_glyph_id"]:
                APP_STATE["next_glyph_id"] = numeric + 1
        except ValueError:
            pass
    else:
        glyph_id = str(APP_STATE["next_glyph_id"])
        APP_STATE["next_glyph_id"] += 1

    node_tag = _new_tag()
    attr_tags: dict[str, int] = {}

    _gid = glyph_id  # captura para closures

    with dpg.node(label=f"[P] {name}", parent=NODE_EDITOR_TAG,
                  pos=[int(pos_x), int(pos_y)], tag=node_tag):

        # Barra de controle: [X] [Abrir]
        ctrl_attr = _new_tag()
        with dpg.node_attribute(tag=ctrl_attr,
                                attribute_type=dpg.mvNode_Attr_Static):
            with dpg.group(horizontal=True):
                dpg.add_button(label="X", width=22, height=18,
                               callback=lambda: remove_procedure(_gid),
                               tag=_new_tag())
                dpg.add_button(label=t("proc_open"), width=50, height=18,
                               callback=lambda: open_procedure_popup(name),
                               tag=_new_tag())

        # Porta de entrada "i" (External Input do workspace pai)
        i_tag = _new_tag()
        attr_tags["i_in"] = i_tag
        with dpg.node_attribute(label="i", tag=i_tag,
                                attribute_type=dpg.mvNode_Attr_Input):
            dpg.add_text("i")

        # Porta de saída "o" (External Output do workspace pai)
        o_tag = _new_tag()
        attr_tags["o_out"] = o_tag
        with dpg.node_attribute(label="o", tag=o_tag,
                                attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("o")

    # Aplica tema azul
    if _PROC_THEME_TAG and dpg.does_item_exist(_PROC_THEME_TAG):
        dpg.bind_item_theme(node_tag, _PROC_THEME_TAG)

    # Registra o mapeamento dos atributos globais (para conexões no canvas principal)
    from gui.canvas import _attr_tag_to_port
    _attr_tag_to_port[i_tag] = (glyph_id, "i_in")
    _attr_tag_to_port[o_tag] = (glyph_id, "o_out")

    # IDs internos para ExtPort
    ext_in_id  = str(_alloc_proc_glyph_id_global())
    ext_out_id = str(_alloc_proc_glyph_id_global())

    proc_state = ProcedureState(
        name=name,
        glyph_id=glyph_id,
        pos_x=pos_x,
        pos_y=pos_y,
        dpg_node_tag=node_tag,
        dpg_attr_tags=attr_tags,
        ext_in_glyph_id=ext_in_id,
        ext_out_glyph_id=ext_out_id,
    )
    APP_STATE["procedures"][name] = proc_state
    APP_STATE["dirty"] = True
    return glyph_id


def _alloc_proc_glyph_id_global() -> int:
    """Aloca um ID global do workspace principal (para evitar colisões)."""
    gid = APP_STATE["next_glyph_id"]
    APP_STATE["next_glyph_id"] += 1
    return gid


# ---------------------------------------------------------------------------
# Popup do editor interno
# ---------------------------------------------------------------------------

def open_procedure_popup(proc_name: str):
    """Abre (ou foca) o popup do editor interno da procedure."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return

    # Se popup já aberto, apenas foca
    if proc.window_tag and dpg.does_item_exist(proc.window_tag):
        dpg.focus_item(proc.window_tag)
        APP_STATE["active_procedure"] = proc_name
        return

    win_tag = _new_tag()
    proc.window_tag = win_tag

    APP_STATE["active_procedure"] = proc_name

    def _on_close():
        _close_procedure_popup(proc_name)

    with dpg.window(
        label=t("proc_editor", name=proc_name),
        tag=win_tag,
        width=900, height=650,
        on_close=_on_close,
        pos=[80, 80],
    ):
        # Barra superior
        with dpg.group(horizontal=True):
            dpg.add_text(t("proc_editor", name=proc_name))
            dpg.add_button(label=t("proc_close_popup"),
                           callback=_on_close)

        dpg.add_separator()

        # Node editor interno
        ne_tag = _new_tag()
        proc.node_editor_tag = ne_tag
        dpg.add_node_editor(
            tag=ne_tag,
            callback=lambda s, d: _on_proc_link_created(proc_name, s, d),
            delink_callback=lambda s, d: _on_proc_link_deleted(proc_name, s, d),
            minimap=True,
            minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
        )

    # Cria nós fixos de ExternalInput e ExternalOutput
    _create_ext_nodes(proc_name)

    # Reconstrói glyphs e links internos já existentes
    _restore_proc_glyphs(proc_name)
    _restore_proc_links(proc_name)


def _close_procedure_popup(proc_name: str):
    """Fecha o popup e limpa referências DPG (links e nós são reconstruídos ao reabrir)."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return

    # Limpa mapeamento de link DPG para este proc
    for tag in [t for t, (pn, _) in list(_dpg_link_to_id.items()) if pn == proc_name]:
        _dpg_link_to_id.pop(tag, None)

    # Limpa attr_tag_to_port (serão recriados ao reabrir)
    proc.attr_tag_to_port.clear()

    if proc.window_tag and dpg.does_item_exist(proc.window_tag):
        dpg.delete_item(proc.window_tag)
    proc.window_tag = 0
    proc.node_editor_tag = 0
    if APP_STATE["active_procedure"] == proc_name:
        APP_STATE["active_procedure"] = None


def _create_ext_nodes(proc_name: str):
    """Cria os nós External Input e External Output dentro do popup."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc or not dpg.does_item_exist(proc.node_editor_tag):
        return

    ne = proc.node_editor_tag

    # Tema verde escuro para ExternalInput
    ext_in_theme = _new_tag()
    with dpg.theme(tag=ext_in_theme):
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (20, 90, 40, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (10, 60, 25, 255),
                                category=dpg.mvThemeCat_Nodes)

    # Tema vermelho escuro para ExternalOutput
    ext_out_theme = _new_tag()
    with dpg.theme(tag=ext_out_theme):
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (120, 30, 30, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (80, 15, 15, 255),
                                category=dpg.mvThemeCat_Nodes)

    # ── ExternalInput ────────────────────────────────────────────────────────
    ext_in_node = _new_tag()
    ext_in_out_attr = _new_tag()   # porta de SAÍDA para dentro da procedure

    with dpg.node(label=t("ext_input"), parent=ne,
                  pos=[30, 200], tag=ext_in_node):
        with dpg.node_attribute(label="o", tag=ext_in_out_attr,
                                attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("o")

    dpg.bind_item_theme(ext_in_node, ext_in_theme)

    # ── ExternalOutput ───────────────────────────────────────────────────────
    ext_out_node = _new_tag()
    ext_out_in_attr = _new_tag()   # porta de ENTRADA vinda de dentro da procedure

    with dpg.node(label=t("ext_output"), parent=ne,
                  pos=[650, 200], tag=ext_out_node):
        with dpg.node_attribute(label="o", tag=ext_out_in_attr,
                                attribute_type=dpg.mvNode_Attr_Input):
            dpg.add_text("o")

    dpg.bind_item_theme(ext_out_node, ext_out_theme)

    # Registra no attr_tag_to_port da procedure
    proc.attr_tag_to_port[ext_in_out_attr]  = (proc.ext_in_glyph_id,  "o_out")
    proc.attr_tag_to_port[ext_out_in_attr]  = (proc.ext_out_glyph_id, "o_in")

    # Salva tags para uso posterior (GlyphState sintético para ExtPorts)
    _save_ext_glyph_state(proc, "External Input (1)", proc.ext_in_glyph_id,
                          ext_in_node, {"o_out": ext_in_out_attr}, 30, 200)
    _save_ext_glyph_state(proc, "External Output (1)", proc.ext_out_glyph_id,
                          ext_out_node, {"o_in": ext_out_in_attr}, 650, 200)


def _save_ext_glyph_state(proc: ProcedureState, func: str, glyph_id: str,
                           node_tag: int, attr_tags: dict,
                           pos_x: float, pos_y: float):
    """Registra estado sintético de ExtPort no proc.glyphs."""
    state = GlyphState(
        glyph_id=glyph_id, func=func,
        library="ExtPort",
        pos_x=pos_x, pos_y=pos_y,
        params={},
        dpg_node_tag=node_tag,
        dpg_attr_tags=attr_tags,
    )
    proc.glyphs[glyph_id] = state


def _restore_proc_glyphs(proc_name: str):
    """Reconstrói glyphs internos (exceto ExtPorts) no popup."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return
    # Glyphs com func != ExtPort e que ainda não têm node_tag criado
    for gid, state in list(proc.glyphs.items()):
        if state.library == "ExtPort":
            continue
        # Já está no popup — verificar se existe
        if dpg.does_item_exist(state.dpg_node_tag):
            continue
        # Recria o nó no popup
        _build_proc_glyph_node(proc_name, state.func, state.pos_x, state.pos_y,
                                gid, state.params)


def _restore_proc_links(proc_name: str):
    """Reconstrói links internos a partir de (out_gid, out_key, in_gid, in_key)."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc or not dpg.does_item_exist(proc.node_editor_tag):
        return

    # Limpa mapeamentos DPG antigos deste proc
    for tag in [t for t, (pn, _) in list(_dpg_link_to_id.items()) if pn == proc_name]:
        _dpg_link_to_id.pop(tag, None)

    for link_id, (out_gid, out_key, in_gid, in_key) in proc.links.items():
        out_state = proc.glyphs.get(str(out_gid))
        in_state  = proc.glyphs.get(str(in_gid))
        if not out_state or not in_state:
            continue
        out_attr = out_state.dpg_attr_tags.get(out_key)
        in_attr  = in_state.dpg_attr_tags.get(in_key)
        if out_attr and in_attr and dpg.does_item_exist(out_attr) and dpg.does_item_exist(in_attr):
            try:
                link_tag = _new_tag()
                dpg.add_node_link(out_attr, in_attr,
                                  parent=proc.node_editor_tag, tag=link_tag)
                _dpg_link_to_id[link_tag] = (proc_name, link_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Adicionar glyph dentro de uma procedure
# ---------------------------------------------------------------------------

def add_glyph_to_procedure(proc_name: str, func: str,
                            pos_x: float = 300, pos_y: float = 200,
                            forced_id: str = None,
                            params: dict = None) -> str | None:
    """
    Adiciona um glyph VGL dentro do node_editor de uma procedure.
    Retorna o glyph_id interno ou None em caso de erro.
    """
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return None

    if func not in GLYPH_REGISTRY:
        return None

    # Aloca ID interno
    if forced_id is not None:
        glyph_id = str(forced_id)
        try:
            numeric = int(glyph_id)
            if numeric >= proc.next_glyph_id:
                proc.next_glyph_id = numeric + 1
        except ValueError:
            pass
    else:
        glyph_id = str(proc.next_glyph_id)
        proc.next_glyph_id += 1

    glyph_def = GLYPH_REGISTRY[func]
    merged_params: dict[str, str] = {p.name: p.default for p in glyph_def.params}
    if params:
        merged_params.update({k: str(v) for k, v in params.items()})

    # Se popup está aberto, cria o nó visual
    if proc.node_editor_tag and dpg.does_item_exist(proc.node_editor_tag):
        _build_proc_glyph_node(proc_name, func, pos_x, pos_y, glyph_id, merged_params)
    else:
        # Popup fechado — salva estado para reconstruir quando abrir
        state = GlyphState(
            glyph_id=glyph_id, func=func,
            library=glyph_def.library,
            pos_x=pos_x, pos_y=pos_y,
            params=merged_params,
            dpg_node_tag=0,
            dpg_attr_tags={},
        )
        proc.glyphs[glyph_id] = state

    APP_STATE["dirty"] = True
    return glyph_id


def _build_proc_glyph_node(proc_name: str, func: str,
                             pos_x: float, pos_y: float,
                             glyph_id: str, params: dict):
    """Constrói o widget DPG do glyph dentro do popup da procedure."""
    from gui.canvas import _add_param_widget
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc or not dpg.does_item_exist(proc.node_editor_tag):
        return

    glyph_def = GLYPH_REGISTRY[func]
    node_tag  = _new_tag()
    attr_tags: dict[str, int] = {}

    with dpg.node(label=glyph_def.label, parent=proc.node_editor_tag,
                  pos=[int(pos_x), int(pos_y)], tag=node_tag):

        # Barra de controle
        ctrl_attr = _new_tag()
        _gid = glyph_id
        _pname = proc_name
        with dpg.node_attribute(tag=ctrl_attr,
                                attribute_type=dpg.mvNode_Attr_Static):
            with dpg.group(horizontal=True):
                dpg.add_button(label="X", width=22, height=18,
                               callback=lambda: _remove_proc_glyph(_pname, _gid),
                               tag=_new_tag())
                dpg.add_text(f"[{glyph_id}]")

        # Portas de entrada
        seen_inputs: set[str] = set()
        for port in glyph_def.ports:
            if port.kind != "input":
                continue
            port_key = port.name + "_in"
            if port_key in seen_inputs:
                continue
            seen_inputs.add(port_key)
            tag = _new_tag()
            attr_tags[port_key] = tag
            proc.attr_tag_to_port[tag] = (glyph_id, port_key)
            with dpg.node_attribute(label=port.name, tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text(port.name)

        # Parâmetros
        for param in glyph_def.params:
            if param.type == "hidden":
                continue
            tag = _new_tag()
            attr_tags["param_" + param.name] = tag
            proc.attr_tag_to_port[tag] = (glyph_id, "param_" + param.name)
            with dpg.node_attribute(label=f"##{param.name}_{glyph_id}_p",
                                    tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Static):
                _add_proc_param_widget(param, proc_name, glyph_id,
                                       params.get(param.name, ""), params)

        # Portas de saída
        seen_outputs: set[str] = set()
        for port in glyph_def.ports:
            if port.kind != "output":
                continue
            port_key = port.name + "_out"
            if port_key in seen_outputs:
                continue
            seen_outputs.add(port_key)
            tag = _new_tag()
            attr_tags[port_key] = tag
            proc.attr_tag_to_port[tag] = (glyph_id, port_key)
            with dpg.node_attribute(label=port.name, tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text(port.name)

    # Registra estado
    state = GlyphState(
        glyph_id=glyph_id, func=func,
        library=glyph_def.library,
        pos_x=pos_x, pos_y=pos_y,
        params=params,
        dpg_node_tag=node_tag,
        dpg_attr_tags=attr_tags,
    )
    proc.glyphs[glyph_id] = state


def _add_proc_param_widget(param: ParamDef, proc_name: str, glyph_id: str,
                            value: str, all_params: dict):
    """Cria widget de parâmetro dentro do editor de procedure."""
    widget_tag = _new_tag()

    def _on_change(s, v):
        proc = APP_STATE["procedures"].get(proc_name)
        if proc:
            state = proc.glyphs.get(glyph_id)
            if state:
                state.params[param.name] = str(v)
                APP_STATE["dirty"] = True

    if param.type == "file":
        dpg.add_text(param.label + ":")
        dpg.add_input_text(tag=widget_tag, default_value=value, width=140,
                           callback=_on_change)
    elif param.type == "int":
        try:
            int_val = int(value) if value else 0
        except (ValueError, TypeError):
            int_val = 0
        dpg.add_input_int(label=param.label, tag=widget_tag, width=80,
                          default_value=int_val, callback=_on_change)
    elif param.type == "float":
        try:
            float_val = float(value) if value else 0.0
        except (ValueError, TypeError):
            float_val = 0.0
        dpg.add_input_float(label=param.label, tag=widget_tag, width=80,
                            default_value=float_val, callback=_on_change)
    elif param.type == "bool":
        try:
            bool_val = bool(int(value)) if value else False
        except (ValueError, TypeError):
            bool_val = False
        dpg.add_checkbox(label=param.label, tag=widget_tag,
                         default_value=bool_val, callback=_on_change)
    elif param.type == "array":
        dpg.add_text(param.label + ":")
        dpg.add_input_text(tag=widget_tag, default_value=value,
                           width=160, callback=_on_change)
    else:
        dpg.add_input_text(label=param.label, tag=widget_tag,
                           default_value=value, width=120, callback=_on_change)


def _remove_proc_glyph(proc_name: str, glyph_id: str):
    """Remove um glyph do editor interno da procedure."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return
    state = proc.glyphs.get(glyph_id)
    if not state or state.library == "ExtPort":
        return

    # Remove links conectados (filtra por glyph_id nas tuplas)
    gid_str = str(glyph_id)
    link_ids_to_remove = [
        lid for lid, (og, _, ig, _) in proc.links.items()
        if str(og) == gid_str or str(ig) == gid_str
    ]
    for lid in link_ids_to_remove:
        # Remove DPG link tag(s) correspondentes
        for dpg_tag, (pn, stored_lid) in list(_dpg_link_to_id.items()):
            if pn == proc_name and stored_lid == lid:
                if dpg.does_item_exist(dpg_tag):
                    dpg.delete_item(dpg_tag)
                _dpg_link_to_id.pop(dpg_tag, None)
        proc.links.pop(lid, None)

    # Remove do attr_tag_to_port
    for tag in state.dpg_attr_tags.values():
        proc.attr_tag_to_port.pop(tag, None)

    # Remove nó
    if dpg.does_item_exist(state.dpg_node_tag):
        dpg.delete_item(state.dpg_node_tag)

    del proc.glyphs[glyph_id]
    APP_STATE["dirty"] = True


# ---------------------------------------------------------------------------
# Links internos
# ---------------------------------------------------------------------------

def _on_proc_link_created(proc_name: str, sender, app_data):
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return
    out_attr, in_attr = app_data

    out_info = proc.attr_tag_to_port.get(out_attr)
    in_info  = proc.attr_tag_to_port.get(in_attr)
    if not out_info or not in_info:
        return

    out_gid, out_key = out_info
    in_gid,  in_key  = in_info

    link_id = proc.next_link_id
    proc.next_link_id += 1
    proc.links[link_id] = (str(out_gid), out_key, str(in_gid), in_key)

    link_tag = _new_tag()
    dpg.add_node_link(out_attr, in_attr, parent=proc.node_editor_tag, tag=link_tag)
    _dpg_link_to_id[link_tag] = (proc_name, link_id)
    APP_STATE["dirty"] = True


def _on_proc_link_deleted(proc_name: str, sender, app_data):
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return
    link_tag = app_data
    mapping = _dpg_link_to_id.pop(link_tag, None)
    if mapping:
        _, link_id = mapping
        proc.links.pop(link_id, None)
    if dpg.does_item_exist(link_tag):
        dpg.delete_item(link_tag)
    APP_STATE["dirty"] = True


def connect_ports_in_procedure(proc_name: str, out_gid: str, out_port: str,
                                in_gid: str, in_port: str) -> bool:
    """
    Cria uma conexão programática entre dois nós dentro de uma procedure.
    Usado pelo load_wksp ao reconstruir.
    """
    from gui.log_panel import append_log
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return False

    out_gid_s = str(out_gid)
    in_gid_s  = str(in_gid)
    out_state = proc.glyphs.get(out_gid_s)
    in_state  = proc.glyphs.get(in_gid_s)
    if not out_state or not in_state:
        append_log(f"[procedure] connect: glyph não encontrado {out_gid} ou {in_gid}")
        return False

    # Resolve chaves de porta — tenta variações para ExtPorts
    def _resolve_out(state, port):
        for key in (port + "_out", "o_out"):
            if key in state.dpg_attr_tags:
                return key
        return None

    def _resolve_in(state, port):
        for key in (port + "_in", "o_in"):
            if key in state.dpg_attr_tags:
                return key
        return None

    out_key = _resolve_out(out_state, out_port)
    in_key  = _resolve_in(in_state, in_port)
    if out_key is None or in_key is None:
        return False

    # Salva link como (glyph_id, port_key) — estável entre aberturas do popup
    link_id = proc.next_link_id
    proc.next_link_id += 1
    proc.links[link_id] = (out_gid_s, out_key, in_gid_s, in_key)

    # Se popup está aberto, cria o link visual agora
    if dpg.does_item_exist(proc.node_editor_tag):
        out_attr = out_state.dpg_attr_tags.get(out_key)
        in_attr  = in_state.dpg_attr_tags.get(in_key)
        if out_attr and in_attr and dpg.does_item_exist(out_attr) and dpg.does_item_exist(in_attr):
            try:
                link_tag = _new_tag()
                dpg.add_node_link(out_attr, in_attr,
                                  parent=proc.node_editor_tag, tag=link_tag)
                _dpg_link_to_id[link_tag] = (proc_name, link_id)
            except Exception as e:
                append_log(f"[procedure] link error: {e}")

    return True


# ---------------------------------------------------------------------------
# Remover procedure inteira
# ---------------------------------------------------------------------------

def remove_procedure(proc_id_or_name):
    """
    Remove a procedure do canvas principal e fecha o popup.
    Aceita glyph_id (str) ou nome da procedure.
    """
    from gui.canvas import _attr_tag_to_port
    # Resolve proc_name
    proc_name = None
    if proc_id_or_name in APP_STATE["procedures"]:
        proc_name = proc_id_or_name
    else:
        # Busca por glyph_id
        for name, p in APP_STATE["procedures"].items():
            if p.glyph_id == str(proc_id_or_name):
                proc_name = name
                break

    if not proc_name:
        return

    proc = APP_STATE["procedures"][proc_name]

    # Limpa mapeamento de links DPG internos
    for tag in [t for t, (pn, _) in list(_dpg_link_to_id.items()) if pn == proc_name]:
        _dpg_link_to_id.pop(tag, None)

    # Remove links no canvas principal conectados a este nó
    my_tags = set(proc.dpg_attr_tags.values())
    links_to_remove = [
        lid for lid, (o, i) in APP_STATE["links"].items()
        if o in my_tags or i in my_tags
    ]
    for lid in links_to_remove:
        if dpg.does_item_exist(lid):
            dpg.delete_item(lid)
        APP_STATE["links"].pop(lid, None)

    # Remove attr_tag_to_port globais
    for tag in my_tags:
        _attr_tag_to_port.pop(tag, None)

    # Remove nó no canvas principal
    if dpg.does_item_exist(proc.dpg_node_tag):
        dpg.delete_item(proc.dpg_node_tag)

    # Fecha popup se aberto
    if proc.window_tag and dpg.does_item_exist(proc.window_tag):
        dpg.delete_item(proc.window_tag)

    del APP_STATE["procedures"][proc_name]
    if APP_STATE["active_procedure"] == proc_name:
        APP_STATE["active_procedure"] = None

    APP_STATE["dirty"] = True


# ---------------------------------------------------------------------------
# Posições dos nós internos (para salvar)
# ---------------------------------------------------------------------------

def get_all_procedure_positions(proc_name: str) -> dict[str, tuple[float, float]]:
    """Retorna as posições atuais dos nós internos (glyph_id → (x, y))."""
    proc = APP_STATE["procedures"].get(proc_name)
    if not proc:
        return {}
    result = {}
    for gid, state in proc.glyphs.items():
        if dpg.does_item_exist(state.dpg_node_tag):
            pos = dpg.get_item_pos(state.dpg_node_tag)
            result[gid] = (pos[0], pos[1])
        else:
            result[gid] = (state.pos_x, state.pos_y)
    return result


# ---------------------------------------------------------------------------
# Dialog "Nova Procedure"
# ---------------------------------------------------------------------------

def show_new_procedure_dialog():
    """Exibe dialog para o usuário nomear uma nova procedure."""
    popup_tag = _new_tag()
    input_tag = _new_tag()
    error_tag = _new_tag()

    with dpg.window(label=t("proc_new"), modal=True, tag=popup_tag,
                    width=320, no_resize=True, pos=[500, 350]):
        dpg.add_text(t("proc_name_label"))
        dpg.add_input_text(tag=input_tag, hint=t("proc_name_hint"), width=280)
        dpg.add_text("", tag=error_tag, color=(255, 100, 100, 255))
        dpg.add_separator()

        def _on_create():
            name = dpg.get_value(input_tag).strip()
            if not name:
                dpg.set_value(error_tag, t("proc_name_empty"))
                return
            if name in APP_STATE["procedures"]:
                dpg.set_value(error_tag, t("proc_name_dup"))
                return
            dpg.delete_item(popup_tag)
            add_procedure_to_canvas(name)

        with dpg.group(horizontal=True):
            dpg.add_button(label=t("proc_create"), width=120, callback=_on_create)
            dpg.add_button(label=t("proc_cancel"), width=120,
                           callback=lambda: dpg.delete_item(popup_tag))
