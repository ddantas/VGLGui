import dearpygui.dearpygui as dpg
from gui.app import APP_STATE, GlyphState
from gui.glyph_registry import GLYPH_REGISTRY, ParamDef

NODE_EDITOR_TAG = 100

# Cores por status — estilo Cantata
_STATUS_COLORS: dict[str, tuple] = {
    "ready":   (70,  70,  70,  255),
    "running": (20,  20,  20,  255),
    "done":    (210, 210, 210, 255),
    "error":   (160, 30,  30,  255),
}

# Temas por status — criados uma vez e reutilizados
_status_themes: dict[str, int] = {}

# Mapeamento attr_tag → (glyph_id, port_key) — usado pelo wksp_io
_attr_tag_to_port: dict[int, tuple[str, str]] = {}

# Controle do próximo ID de tag DPG
_tag_counter = 2000


def _new_tag() -> int:
    global _tag_counter
    _tag_counter += 1
    return _tag_counter


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_canvas():
    dpg.add_node_editor(
        tag=NODE_EDITOR_TAG,
        callback=on_link_created,
        delink_callback=on_link_deleted,
        minimap=True,
        minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
    )
    _build_status_themes()


def _build_status_themes():
    for status, (r, g, b, a) in _STATUS_COLORS.items():
        theme_tag = _new_tag()
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,
                                    (r, g, b, a), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,
                                    (min(r+20,255), min(g+20,255), min(b+20,255), a),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected,
                                    (min(r+40,255), min(g+40,255), min(b+40,255), a),
                                    category=dpg.mvThemeCat_Nodes)
        _status_themes[status] = theme_tag


# ---------------------------------------------------------------------------
# Adicionar Glyph ao canvas
# ---------------------------------------------------------------------------

def add_glyph_to_canvas(func: str, pos_x: float = 100, pos_y: float = 100,
                        forced_id: str = None, params: dict = None) -> str:
    if func not in GLYPH_REGISTRY:
        print(f"[canvas] Função desconhecida: {func}")
        return None

    glyph_def = GLYPH_REGISTRY[func]

    # Determina o ID do glyph
    if forced_id is not None:
        glyph_id = str(forced_id)
        # Garante que next_glyph_id não vai colidir
        try:
            numeric = int(glyph_id)
            if numeric >= APP_STATE["next_glyph_id"]:
                APP_STATE["next_glyph_id"] = numeric + 1
        except ValueError:
            pass
    else:
        glyph_id = str(APP_STATE["next_glyph_id"])
        APP_STATE["next_glyph_id"] += 1

    # Mescla parâmetros: defaults do registry sobrescritos pelos fornecidos
    merged_params: dict[str, str] = {p.name: p.default for p in glyph_def.params}
    if params:
        merged_params.update({k: str(v) for k, v in params.items()})

    node_tag = _new_tag()
    attr_tags: dict[str, int] = {}

    with dpg.node(label=glyph_def.label, parent=NODE_EDITOR_TAG,
                  pos=[int(pos_x), int(pos_y)], tag=node_tag):

        # ── Barra de controle (Delete / Expand / Info + ID) ─────────────────
        ctrl_attr = _new_tag()
        with dpg.node_attribute(tag=ctrl_attr,
                                attribute_type=dpg.mvNode_Attr_Static):
            _gid = glyph_id   # captura para closures
            with dpg.group(horizontal=True):
                dpg.add_button(label="X", width=22, height=18,
                               callback=lambda: remove_glyph(_gid),
                               tag=_new_tag())
                dpg.add_button(label="=", width=22, height=18,
                               callback=lambda: toggle_expand(_gid),
                               tag=_new_tag())
                dpg.add_button(label="i", width=22, height=18,
                               callback=lambda: show_info(_gid),
                               tag=_new_tag())
                dpg.add_text(f"[{glyph_id}]")

        # ── Portas de entrada ────────────────────────────────────────────────
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
            _attr_tag_to_port[tag] = (glyph_id, port_key)

            with dpg.node_attribute(label=port.name, tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text(port.name)

        # ── Parâmetros (inicialmente visíveis; toggle_expand os oculta) ──────
        for param in glyph_def.params:
            tag = _new_tag()
            attr_tags["param_" + param.name] = tag
            _attr_tag_to_port[tag] = (glyph_id, "param_" + param.name)

            with dpg.node_attribute(label=f"##{param.name}_{glyph_id}",
                                    tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Static):
                _add_param_widget(param, glyph_id, merged_params.get(param.name, ""))

        # ── Portas de saída ──────────────────────────────────────────────────
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
            _attr_tag_to_port[tag] = (glyph_id, port_key)

            with dpg.node_attribute(label=port.name, tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text(port.name)

    # Registra no estado global
    state = GlyphState(
        glyph_id=glyph_id,
        func=func,
        library=glyph_def.library,
        pos_x=pos_x,
        pos_y=pos_y,
        params=merged_params,
        dpg_node_tag=node_tag,
        dpg_attr_tags=attr_tags,
    )
    APP_STATE["glyphs"][glyph_id] = state
    APP_STATE["dirty"] = True

    # Aplica tema "ready"
    set_glyph_status(glyph_id, "ready")
    return glyph_id


def _add_param_widget(param: ParamDef, glyph_id: str, value: str):
    """Cria o widget adequado para cada tipo de parâmetro."""
    widget_tag = _new_tag()
    label = param.label

    def _on_change(s, v):
        state = APP_STATE["glyphs"].get(glyph_id)
        if state:
            state.params[param.name] = str(v)
            APP_STATE["dirty"] = True

    if param.type == "file":
        dpg.add_text(label + ":", tag=_new_tag())
        dpg.add_input_text(tag=widget_tag, default_value=value,
                           width=160, callback=_on_change)
        dpg.add_button(label="...", width=25, tag=_new_tag(),
                       callback=lambda: _open_file_dialog(widget_tag, glyph_id, param.name))
    elif param.type == "int":
        dpg.add_input_int(label=label, tag=widget_tag, width=80,
                          default_value=int(value) if value else 0,
                          callback=_on_change)
    elif param.type == "float":
        dpg.add_input_float(label=label, tag=widget_tag, width=80,
                            default_value=float(value) if value else 0.0,
                            callback=_on_change)
    elif param.type == "bool":
        dpg.add_checkbox(label=label, tag=widget_tag,
                         default_value=bool(int(value)) if value else False,
                         callback=_on_change)
    elif param.type == "array":
        dpg.add_text(label + ":", tag=_new_tag())
        dpg.add_input_text(tag=widget_tag, default_value=value,
                           width=190, callback=_on_change)
    else:  # "text"
        dpg.add_input_text(label=label, tag=widget_tag,
                           default_value=value, width=120,
                           callback=_on_change)


def _open_file_dialog(target_tag: int, glyph_id: str, param_name: str):
    """Abre um file dialog nativo do Dear PyGui."""
    def _on_select(s, app_data):
        path = app_data.get("file_path_name", "")
        dpg.set_value(target_tag, path)
        state = APP_STATE["glyphs"].get(glyph_id)
        if state:
            state.params[param_name] = path
            APP_STATE["dirty"] = True

    with dpg.file_dialog(
        label="Selecionar arquivo",
        modal=True,
        width=600, height=400,
        callback=_on_select,
        tag=_new_tag(),
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".png", color=(0, 255, 0, 255))
        dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
        dpg.add_file_extension(".tiff", color=(0, 200, 255, 255))
        dpg.add_file_extension(".wksp", color=(255, 200, 0, 255))


# ---------------------------------------------------------------------------
# Remover Glyph
# ---------------------------------------------------------------------------

def remove_glyph(glyph_id: str):
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return

    # Remove todos os links conectados a este nó
    my_tags = set(state.dpg_attr_tags.values())
    links_to_remove = [
        lid for lid, (out_t, in_t) in APP_STATE["links"].items()
        if out_t in my_tags or in_t in my_tags
    ]
    for lid in links_to_remove:
        if dpg.does_item_exist(lid):
            dpg.delete_item(lid)
        del APP_STATE["links"][lid]

    # Remove tags do mapeamento global
    for tag in my_tags:
        _attr_tag_to_port.pop(tag, None)

    # Remove o nó do canvas
    if dpg.does_item_exist(state.dpg_node_tag):
        dpg.delete_item(state.dpg_node_tag)

    del APP_STATE["glyphs"][glyph_id]
    APP_STATE["dirty"] = True


# ---------------------------------------------------------------------------
# Expand / Collapse parâmetros
# ---------------------------------------------------------------------------

def toggle_expand(glyph_id: str):
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return
    for key, tag in state.dpg_attr_tags.items():
        if key.startswith("param_"):
            if dpg.does_item_exist(tag):
                shown = dpg.is_item_shown(tag)
                dpg.show_item(tag) if not shown else dpg.hide_item(tag)


# ---------------------------------------------------------------------------
# Info popup
# ---------------------------------------------------------------------------

def show_info(glyph_id: str):
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return
    glyph_def = GLYPH_REGISTRY.get(state.func)
    popup_tag = _new_tag()

    with dpg.window(label=f"Info — {state.func}", modal=True,
                    tag=popup_tag, width=350, no_resize=True,
                    pos=[300, 300]):
        dpg.add_text(f"Função:    {state.func}")
        dpg.add_text(f"Biblioteca: {state.library}")
        dpg.add_text(f"Categoria: {glyph_def.category if glyph_def else '?'}")
        dpg.add_separator()
        if glyph_def:
            dpg.add_text("Portas:")
            for p in glyph_def.ports:
                req = "" if p.required else " (opcional)"
                dpg.add_text(f"  [{p.kind}] {p.name}{req}")
            if glyph_def.params:
                dpg.add_text("Parâmetros:")
                for p in glyph_def.params:
                    dpg.add_text(f"  {p.name} ({p.type}) default={p.default!r}")
        dpg.add_separator()
        dpg.add_button(label="Fechar",
                       callback=lambda: dpg.delete_item(popup_tag))


# ---------------------------------------------------------------------------
# Status visual (Cantata style)
# ---------------------------------------------------------------------------

def set_glyph_status(glyph_id: str, status: str):
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return
    state.status = status
    theme = _status_themes.get(status)
    if theme and dpg.does_item_exist(state.dpg_node_tag):
        dpg.bind_item_theme(state.dpg_node_tag, theme)


# ---------------------------------------------------------------------------
# Callbacks de conexão
# ---------------------------------------------------------------------------

def on_link_created(sender, app_data):
    out_attr, in_attr = app_data
    link_tag = _new_tag()
    dpg.add_node_link(out_attr, in_attr, parent=NODE_EDITOR_TAG, tag=link_tag)
    APP_STATE["links"][link_tag] = (out_attr, in_attr)
    APP_STATE["dirty"] = True


def on_link_deleted(sender, app_data):
    link_tag = app_data
    if link_tag in APP_STATE["links"]:
        del APP_STATE["links"][link_tag]
    if dpg.does_item_exist(link_tag):
        dpg.delete_item(link_tag)
    APP_STATE["dirty"] = True


# ---------------------------------------------------------------------------
# Conectar portas programaticamente (usado no load)
# ---------------------------------------------------------------------------

def connect_ports(out_gid: str, out_port: str, in_gid: str, in_port: str):
    """
    Cria uma conexão entre out_gid:out_port → in_gid:in_port.
    port names sem sufixo (_out/_in) — a função adiciona.
    """
    out_state = APP_STATE["glyphs"].get(str(out_gid))
    in_state  = APP_STATE["glyphs"].get(str(in_gid))
    if not out_state or not in_state:
        return

    out_attr = out_state.dpg_attr_tags.get(out_port + "_out")
    in_attr  = in_state.dpg_attr_tags.get(in_port + "_in")

    if out_attr is None or in_attr is None:
        print(f"[canvas] connect_ports: porta não encontrada "
              f"{out_gid}:{out_port}_out → {in_gid}:{in_port}_in")
        return

    link_tag = _new_tag()
    dpg.add_node_link(out_attr, in_attr, parent=NODE_EDITOR_TAG, tag=link_tag)
    APP_STATE["links"][link_tag] = (out_attr, in_attr)


# ---------------------------------------------------------------------------
# Limpar canvas
# ---------------------------------------------------------------------------

def clear_canvas():
    for glyph_id in list(APP_STATE["glyphs"].keys()):
        remove_glyph(glyph_id)
    # Garante limpeza de qualquer link residual
    for lid in list(APP_STATE["links"].keys()):
        if dpg.does_item_exist(lid):
            dpg.delete_item(lid)
    APP_STATE["links"].clear()
    APP_STATE["glyphs"].clear()
    _attr_tag_to_port.clear()
    APP_STATE["next_glyph_id"] = 1
    APP_STATE["dirty"] = False


# ---------------------------------------------------------------------------
# Posições atuais dos nós (para salvar)
# ---------------------------------------------------------------------------

def get_all_node_positions() -> dict[str, tuple[float, float]]:
    result = {}
    for glyph_id, state in APP_STATE["glyphs"].items():
        if dpg.does_item_exist(state.dpg_node_tag):
            pos = dpg.get_item_pos(state.dpg_node_tag)
            result[glyph_id] = (pos[0], pos[1])
    return result


# ---------------------------------------------------------------------------
# Expor mapeamento para wksp_io
# ---------------------------------------------------------------------------

def get_attr_tag_to_port() -> dict[int, tuple[str, str]]:
    return _attr_tag_to_port
