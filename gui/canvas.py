import threading
import dearpygui.dearpygui as dpg
from gui.app import APP_STATE, GlyphState
from gui.glyph_registry import GLYPH_REGISTRY, ParamDef
from gui.i18n import t

NODE_EDITOR_TAG = 0  # definido em setup_canvas
_zoom_level     = 1.0
_ctrl_held      = False

# Cores por status — estilo Cantata
_STATUS_COLORS: dict[str, tuple] = {
    "ready":   (70,  70,  70,  255),
    "running": (180, 130, 20,  255),   # amarelo escuro — em execução
    "done":    (30,  90,  45,  255),   # verde escuro — concluído
    "error":   (160, 30,  30,  255),   # vermelho — erro
}

# Temas por status — criados uma vez e reutilizados
_status_themes: dict[str, int] = {}
_category_themes: dict[str, int] = {}
_use_category_colors: bool = True

# Mapeamento attr_tag → (glyph_id, port_key) — usado pelo wksp_io
_attr_tag_to_port: dict[int, tuple[str, str]] = {}

# Fila thread-safe para atualizações de status vindas do runner (thread de fundo)
_status_queue: list[tuple[str, str]] = []
_status_lock  = threading.Lock()

# Fila thread-safe para previews de imagem nos nós
_preview_queue: list[tuple[str, str]] = []   # (glyph_id, image_path)
_preview_lock  = threading.Lock()

def _new_tag() -> int:
    return dpg.generate_uuid()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_canvas():
    global NODE_EDITOR_TAG
    NODE_EDITOR_TAG = dpg.add_node_editor(
        callback=on_link_created,
        delink_callback=on_link_deleted,
        minimap=True,
        minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
    )
    _build_status_themes()
    _build_category_themes()
    from gui.procedure_canvas import init_procedure_theme
    init_procedure_theme()


def bind_canvas_handlers():
    """Registra handlers globais de teclado e scroll. Chamar de app.py fora de qualquer contexto de janela."""
    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=_on_mouse_wheel)
        dpg.add_key_press_handler(callback=_on_key_press)
        dpg.add_key_release_handler(callback=_on_key_release)
        dpg.add_mouse_click_handler(button=1, callback=_on_right_click)


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


def _build_category_themes():
    from gui.glyph_registry import CATEGORY_COLORS
    def _clamp(v: int) -> int:
        return max(0, min(255, v))
    for category, (r, g, b) in CATEGORY_COLORS.items():
        theme_tag = _new_tag()
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,
                                    (r, g, b, 200), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,
                                    (_clamp(r+20), _clamp(g+20), _clamp(b+20), 220),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected,
                                    (_clamp(r+40), _clamp(g+40), _clamp(b+40), 240),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar,
                                    (_clamp(r-15), _clamp(g-15), _clamp(b-15), 220),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,
                                    (r, g, b, 240),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,
                                    (_clamp(r+25), _clamp(g+25), _clamp(b+25), 255),
                                    category=dpg.mvThemeCat_Nodes)
        _category_themes[category] = theme_tag


# ---------------------------------------------------------------------------
# Zoom (Ctrl + scroll)
# ---------------------------------------------------------------------------

def _on_key_press(sender, app_data):
    global _ctrl_held
    if app_data in (dpg.mvKey_LControl, dpg.mvKey_RControl):
        _ctrl_held = True


def _on_key_release(sender, app_data):
    global _ctrl_held
    if app_data in (dpg.mvKey_LControl, dpg.mvKey_RControl):
        _ctrl_held = False


def _on_mouse_wheel(sender, app_data):
    if not _ctrl_held:
        return
    zoom_canvas(1 if app_data > 0 else -1)


def zoom_canvas(direction: int):
    """Escala posições de todos os nós a partir do centro do canvas."""
    global _zoom_level
    old_zoom = _zoom_level
    factor = 1.15 if direction > 0 else (1.0 / 1.15)
    _zoom_level = max(0.1, min(5.0, _zoom_level * factor))
    if abs(_zoom_level - old_zoom) < 1e-9:
        return

    ratio = _zoom_level / old_zoom
    try:
        sz = dpg.get_item_rect_size("canvas_win")
        cx, cy = sz[0] / 2.0, sz[1] / 2.0
    except Exception:
        cx, cy = 400.0, 300.0

    for state in APP_STATE["glyphs"].values():
        if dpg.does_item_exist(state.dpg_node_tag):
            x, y = dpg.get_item_pos(state.dpg_node_tag)
            dpg.set_item_pos(state.dpg_node_tag,
                             [int(cx + (x - cx) * ratio),
                              int(cy + (y - cy) * ratio)])

    for proc in APP_STATE["procedures"].values():
        if dpg.does_item_exist(proc.dpg_node_tag):
            x, y = dpg.get_item_pos(proc.dpg_node_tag)
            dpg.set_item_pos(proc.dpg_node_tag,
                             [int(cx + (x - cx) * ratio),
                              int(cy + (y - cy) * ratio)])


# ---------------------------------------------------------------------------
# Adicionar Glyph ao canvas
# ---------------------------------------------------------------------------

def add_glyph_to_canvas(func: str, pos_x: float = 100, pos_y: float = 100,
                        forced_id: str = None, params: dict = None) -> str:
    if func not in GLYPH_REGISTRY:
        print(f"[canvas] Unknown function: {func}")
        return None

    # Salva estado para undo apenas quando adicionado pelo usuário (sem forced_id)
    if forced_id is None:
        from gui.history import push_undo
        push_undo()

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
            if param.type == "hidden":
                continue   # gerenciado internamente pelo widget strel
            tag = _new_tag()
            attr_tags["param_" + param.name] = tag
            _attr_tag_to_port[tag] = (glyph_id, "param_" + param.name)

            with dpg.node_attribute(label=f"##{param.name}_{glyph_id}",
                                    tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Static):
                _add_param_widget(param, glyph_id,
                                  merged_params.get(param.name, ""),
                                  merged_params)

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


def _gen_strel(shape: str, w: int, h: int, z: int = 1) -> list[int]:
    """Gera array de elemento estruturante para morfologia."""
    import math
    w, h, z = max(1, w), max(1, h), max(1, z)
    if shape in ("Rectangle", "Box"):
        return [1] * (w * h * z)
    elif shape == "Cross":
        arr = []
        for y in range(h):
            for x in range(w):
                arr.append(1 if (x == w // 2 or y == h // 2) else 0)
        return arr * z
    elif shape == "Diamond":
        arr = []
        cx, cy = w // 2, h // 2
        r = min(cx, cy)
        for y in range(h):
            for x in range(w):
                arr.append(1 if abs(x - cx) + abs(y - cy) <= r else 0)
        return arr * z
    elif shape == "Disc":
        arr = []
        cx, cy = (w - 1) / 2, (h - 1) / 2
        r = min(cx, cy)
        for y in range(h):
            for x in range(w):
                arr.append(1 if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r + 0.5 else 0)
        return arr * z
    return [1] * (w * h * z)


def _add_param_widget(param: ParamDef, glyph_id: str, value: str,
                      all_params: dict = None):
    """Cria o widget adequado para cada tipo de parâmetro."""
    widget_tag = _new_tag()
    label = param.label

    def _on_change(s, v):
        state = APP_STATE["glyphs"].get(glyph_id)
        if state:
            state.params[param.name] = str(v)
            APP_STATE["dirty"] = True

    if param.type == "strel" or param.type == "strel3d":
        _add_strel_widget(param, glyph_id, all_params or {},
                          has_z=(param.type == "strel3d"))
        return

    if param.type == "file":
        dpg.add_text(label + ":", tag=_new_tag())
        dpg.add_input_text(tag=widget_tag, default_value=value,
                           width=160, readonly=True)
        dpg.add_button(label="...", width=25, tag=_new_tag(),
                       callback=lambda: _open_file_dialog(widget_tag, glyph_id, param.name))
    elif param.type == "folder":
        dpg.add_text(label + ":", tag=_new_tag())
        dpg.add_input_text(tag=widget_tag, default_value=value,
                           width=160, readonly=True)
        dpg.add_button(label="dir", width=28, tag=_new_tag(),
                       callback=lambda: _open_folder_dialog(widget_tag, glyph_id, param.name))
    elif param.type == "int":
        try:
            int_val = int(value) if value else 0
        except (ValueError, TypeError):
            int_val = 0
        dpg.add_input_int(label=label, tag=widget_tag, width=80,
                          default_value=int_val,
                          callback=_on_change)
    elif param.type == "float":
        try:
            float_val = float(value) if value else 0.0
        except (ValueError, TypeError):
            float_val = 0.0
        dpg.add_input_float(label=label, tag=widget_tag, width=80,
                            default_value=float_val,
                            callback=_on_change)
    elif param.type == "bool":
        try:
            bool_val = bool(int(value)) if value else False
        except (ValueError, TypeError):
            bool_val = False
        dpg.add_checkbox(label=label, tag=widget_tag,
                         default_value=bool_val,
                         callback=_on_change)
    elif param.type == "array":
        dpg.add_text(label + ":", tag=_new_tag())
        dpg.add_input_text(tag=widget_tag, default_value=value,
                           width=190, callback=_on_change)
    else:  # "text"
        dpg.add_input_text(label=label, tag=widget_tag,
                           default_value=value, width=120,
                           callback=_on_change)


def _add_strel_widget(param: ParamDef, glyph_id: str, all_params: dict,
                      has_z: bool = False):
    """Widget gerador de elemento estruturante morfológico."""
    shapes = ["Rectangle", "Cross", "Diamond", "Disc"] if not has_z else ["Box"]

    def _safe_int(d, key, default=3):
        try:
            return max(1, int(d.get(key, default)))
        except (ValueError, TypeError):
            return default

    w_init = _safe_int(all_params, "window_size_x")
    h_init = _safe_int(all_params, "window_size_y")
    z_init = _safe_int(all_params, "window_size_z") if has_z else 3

    shape_tag  = _new_tag()
    w_tag      = _new_tag()
    h_tag      = _new_tag()
    z_tag      = _new_tag() if has_z else None
    status_tag = _new_tag()

    dpg.add_combo(shapes, tag=shape_tag, default_value=shapes[0], width=130)
    with dpg.group(horizontal=True):
        dpg.add_text("W:", tag=_new_tag())
        dpg.add_input_int(tag=w_tag, default_value=w_init, width=48,
                          min_value=1, max_value=99,
                          min_clamped=True, max_clamped=True)
        dpg.add_text("H:", tag=_new_tag())
        dpg.add_input_int(tag=h_tag, default_value=h_init, width=48,
                          min_value=1, max_value=99,
                          min_clamped=True, max_clamped=True)
        if has_z:
            dpg.add_text("Z:", tag=_new_tag())
            dpg.add_input_int(tag=z_tag, default_value=z_init, width=48,
                              min_value=1, max_value=99,
                              min_clamped=True, max_clamped=True)

    dpg.add_text("", tag=status_tag)   # feedback após gerar

    def _gerar():
        shape = dpg.get_value(shape_tag)
        w     = max(1, dpg.get_value(w_tag))
        h     = max(1, dpg.get_value(h_tag))
        z     = max(1, dpg.get_value(z_tag)) if has_z else 1
        arr   = _gen_strel(shape, w, h, z)
        arr_str = "[" + ",".join(str(v) for v in arr) + "]"
        st = APP_STATE["glyphs"].get(glyph_id)
        if st:
            st.params[param.name]      = arr_str
            st.params["window_size_x"] = str(w)
            st.params["window_size_y"] = str(h)
            if has_z:
                st.params["window_size_z"] = str(z)
            APP_STATE["dirty"] = True
        ones = sum(arr)
        dpg.set_value(status_tag, t("active_px", ones=ones, total=w*h*z))

    dpg.add_button(label=t("generate_se"), tag=_new_tag(),
                   callback=_gerar, width=160)


def _open_file_dialog(target_tag: int, glyph_id: str, param_name: str):
    """Abre um file dialog nativo do Dear PyGui."""
    import os as _os

    def _on_select(s, app_data):
        path = app_data.get("file_path_name", "")
        dpg.set_value(target_tag, path)
        state = APP_STATE["glyphs"].get(glyph_id)
        if state:
            state.params[param_name] = path
            APP_STATE["dirty"] = True

    # Abre no diretório do arquivo atual (se existir), senão no cwd
    current = dpg.get_value(target_tag) or ""
    if _os.path.isfile(current):
        start_dir = _os.path.dirname(current)
    elif _os.path.isdir(current):
        start_dir = current
    else:
        start_dir = _os.getcwd()

    with dpg.file_dialog(
        label=t("select_file"),
        modal=True,
        width=700, height=450,
        callback=_on_select,
        tag=_new_tag(),
        default_path=start_dir,
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".png",  color=(80,  220, 80,  255), custom_text="PNG")
        dpg.add_file_extension(".jpg",  color=(80,  220, 80,  255), custom_text="JPG")
        dpg.add_file_extension(".jpeg", color=(80,  220, 80,  255), custom_text="JPEG")
        dpg.add_file_extension(".tiff", color=(80,  180, 255, 255), custom_text="TIFF")
        dpg.add_file_extension(".pgm",  color=(180, 180, 80,  255), custom_text="PGM")
        dpg.add_file_extension(".ppm",  color=(180, 180, 80,  255), custom_text="PPM")
        dpg.add_file_extension(".wksp", color=(255, 200, 0,   255), custom_text="Workflow")


def _open_folder_dialog(target_tag: int, glyph_id: str, param_name: str):
    """Abre um seletor de pasta (directory_selector=True)."""
    import os as _os

    def _on_select(s, app_data):
        path = app_data.get("file_path_name", "") or app_data.get("current_path", "")
        # DearPyGui retorna o caminho da pasta selecionada
        if _os.path.isfile(path):
            path = _os.path.dirname(path)
        dpg.set_value(target_tag, path)
        state = APP_STATE["glyphs"].get(glyph_id)
        if state:
            state.params[param_name] = path
            APP_STATE["dirty"] = True

    current = dpg.get_value(target_tag) or ""
    start_dir = current if _os.path.isdir(current) else _os.getcwd()

    with dpg.file_dialog(
        label="Selecionar pasta",
        modal=True,
        width=700, height=450,
        callback=_on_select,
        tag=_new_tag(),
        default_path=start_dir,
        directory_selector=True,
    ):
        pass


# ---------------------------------------------------------------------------
# Remover Glyph
# ---------------------------------------------------------------------------

def remove_glyph(glyph_id: str, _record: bool = True):
    if _record:
        from gui.history import push_undo
        push_undo()
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

    # Limpa textura de preview (node_attribute é filho do nó, deletado junto)
    if state.preview_tex_tag and dpg.does_item_exist(state.preview_tex_tag):
        dpg.delete_item(state.preview_tex_tag)

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

    with dpg.window(label=t("info_title", func=state.func), modal=True,
                    tag=popup_tag, width=350, no_resize=True,
                    pos=[300, 300]):
        dpg.add_text(t("func_label", func=state.func))
        dpg.add_text(t("lib_label", lib=state.library))
        dpg.add_text(t("cat_label", cat=glyph_def.category if glyph_def else "?"))
        dpg.add_separator()
        if glyph_def:
            dpg.add_text(t("ports_label"))
            for p in glyph_def.ports:
                req = "" if p.required else t("optional")
                dpg.add_text(f"  [{p.kind}] {p.name}{req}")
            if glyph_def.params:
                dpg.add_text(t("params_label"))
                for p in glyph_def.params:
                    dpg.add_text(f"  {p.name} ({p.type}) default={p.default!r}")
        dpg.add_separator()
        dpg.add_button(label=t("close"),
                       callback=lambda: dpg.delete_item(popup_tag))


# ---------------------------------------------------------------------------
# Status visual (Cantata style)
# ---------------------------------------------------------------------------

def set_glyph_status(glyph_id: str, status: str):
    """Aplica status imediatamente — só chamar da thread principal."""
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return
    state.status = status
    if not dpg.does_item_exist(state.dpg_node_tag):
        return
    if status == "ready":
        # Use category color when idle (if enabled)
        cat_theme = None
        if _use_category_colors:
            glyph_def = GLYPH_REGISTRY.get(state.func)
            cat_theme = (_category_themes.get(glyph_def.category)
                         if glyph_def else None)
        if cat_theme:
            dpg.bind_item_theme(state.dpg_node_tag, cat_theme)
        else:
            theme = _status_themes.get("ready")
            if theme:
                dpg.bind_item_theme(state.dpg_node_tag, theme)
    else:
        theme = _status_themes.get(status)
        if theme:
            dpg.bind_item_theme(state.dpg_node_tag, theme)


def toggle_category_colors():
    """Alterna cores por categoria. Reaplica tema a todos os nós ready."""
    global _use_category_colors
    _use_category_colors = not _use_category_colors
    for gid, state in APP_STATE["glyphs"].items():
        if state.status == "ready":
            set_glyph_status(gid, "ready")


def queue_glyph_status(glyph_id: str, status: str):
    """Thread-safe: enfileira uma atualização de status para aplicar no próximo frame."""
    with _status_lock:
        _status_queue.append((glyph_id, status))


def flush_status_queue():
    """Chamado no render loop da thread principal — aplica atualizações pendentes."""
    if not _status_queue:
        return
    with _status_lock:
        pending = _status_queue.copy()
        _status_queue.clear()
    for glyph_id, status in pending:
        set_glyph_status(glyph_id, status)


# ---------------------------------------------------------------------------
# Callbacks de conexão
# ---------------------------------------------------------------------------

def on_link_created(sender, app_data):
    from gui.history import push_undo
    push_undo()
    out_attr, in_attr = app_data
    link_tag = _new_tag()
    dpg.add_node_link(out_attr, in_attr, parent=NODE_EDITOR_TAG, tag=link_tag)
    APP_STATE["links"][link_tag] = (out_attr, in_attr)
    APP_STATE["dirty"] = True


def on_link_deleted(sender, app_data):
    from gui.history import push_undo
    push_undo()
    link_tag = app_data
    if link_tag in APP_STATE["links"]:
        del APP_STATE["links"][link_tag]
    if dpg.does_item_exist(link_tag):
        dpg.delete_item(link_tag)
    APP_STATE["dirty"] = True


# ---------------------------------------------------------------------------
# Conectar portas programaticamente (usado no load)
# ---------------------------------------------------------------------------

def connect_ports(out_gid: str, out_port: str, in_gid: str, in_port: str) -> bool:
    """
    Cria uma conexão entre out_gid:out_port → in_gid:in_port no canvas principal.
    Suporta glyphs normais e nós de procedure.
    port names sem sufixo (_out/_in) — a função adiciona.
    Retorna True se a conexão foi criada com sucesso.
    """
    from gui.log_panel import append_log

    def _get_attr_tags(gid: str):
        """Retorna dpg_attr_tags de glyph normal ou de nó procedure."""
        state = APP_STATE["glyphs"].get(str(gid))
        if state:
            return state.dpg_attr_tags
        for proc in APP_STATE["procedures"].values():
            if proc.glyph_id == str(gid):
                return proc.dpg_attr_tags
        return None

    out_tags = _get_attr_tags(out_gid)
    in_tags  = _get_attr_tags(in_gid)

    if out_tags is None:
        append_log(f"[canvas] connect_ports: glyph/proc não encontrado {out_gid}")
        return False
    if in_tags is None:
        append_log(f"[canvas] connect_ports: glyph/proc não encontrado {in_gid}")
        return False

    out_attr = out_tags.get(out_port + "_out")
    in_attr  = in_tags.get(in_port + "_in")

    if out_attr is None or in_attr is None:
        append_log(f"[canvas] porta não encontrada: "
                   f"{out_gid}:{out_port}_out -> {in_gid}:{in_port}_in")
        return False

    if not dpg.does_item_exist(out_attr) or not dpg.does_item_exist(in_attr):
        append_log(f"[canvas] DPG item não existe: attr {out_attr} ou {in_attr}")
        return False

    try:
        link_tag = _new_tag()
        dpg.add_node_link(out_attr, in_attr, parent=NODE_EDITOR_TAG, tag=link_tag)
        APP_STATE["links"][link_tag] = (out_attr, in_attr)
        return True
    except Exception as e:
        append_log(f"[canvas] erro ao criar link {out_gid}:{out_port} -> "
                   f"{in_gid}:{in_port}: {e}")
        return False


# ---------------------------------------------------------------------------
# Limpar canvas
# ---------------------------------------------------------------------------

def clear_canvas():
    global _zoom_level
    _zoom_level = 1.0
    from gui.history import clear_history
    clear_history()

    # Remove todas as procedures (fecha popups e nós do canvas principal)
    from gui.procedure_canvas import remove_procedure
    for proc_name in list(APP_STATE["procedures"].keys()):
        remove_procedure(proc_name)
    APP_STATE["active_procedure"] = None

    for glyph_id in list(APP_STATE["glyphs"].keys()):
        remove_glyph(glyph_id, _record=False)
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
# Fit to Screen — ajusta zoom e posição para mostrar todos os nós
# ---------------------------------------------------------------------------

def fit_to_screen():
    """Centraliza e escala todos os nós para caberem no canvas visível."""
    global _zoom_level

    # Coleta posições de todos os nós (glyphs + procedures)
    all_nodes: list[tuple[int, int]] = []
    glyph_tags: list[int] = []
    proc_tags:  list[int] = []

    for state in APP_STATE["glyphs"].values():
        if dpg.does_item_exist(state.dpg_node_tag):
            pos = dpg.get_item_pos(state.dpg_node_tag)
            all_nodes.append((pos[0], pos[1]))
            glyph_tags.append(state.dpg_node_tag)

    for proc in APP_STATE["procedures"].values():
        if dpg.does_item_exist(proc.dpg_node_tag):
            pos = dpg.get_item_pos(proc.dpg_node_tag)
            all_nodes.append((pos[0], pos[1]))
            proc_tags.append(proc.dpg_node_tag)

    if not all_nodes:
        return

    PADDING = 60
    NODE_W, NODE_H = 200, 120  # estimativa de tamanho médio de nó

    xs = [p[0] for p in all_nodes]
    ys = [p[1] for p in all_nodes]
    min_x, max_x = min(xs), max(xs) + NODE_W
    min_y, max_y = min(ys), max(ys) + NODE_H

    content_w = max_x - min_x
    content_h = max_y - min_y
    content_cx = (min_x + max_x) / 2
    content_cy = (min_y + max_y) / 2

    try:
        sz = dpg.get_item_rect_size("canvas_win")
        canvas_w, canvas_h = sz[0], sz[1]
    except Exception:
        canvas_w, canvas_h = 900, 600

    avail_w = max(1, canvas_w - 2 * PADDING)
    avail_h = max(1, canvas_h - 2 * PADDING)

    scale = min(avail_w / max(1, content_w),
                avail_h / max(1, content_h),
                2.0)  # nunca ampliar demais

    target_cx = canvas_w / 2
    target_cy = canvas_h / 2

    for tag, (ox, oy) in zip(glyph_tags + proc_tags,
                              [dpg.get_item_pos(t) for t in glyph_tags + proc_tags]):
        new_x = target_cx + (ox - content_cx) * scale
        new_y = target_cy + (oy - content_cy) * scale
        dpg.set_item_pos(tag, [int(new_x), int(new_y)])

    _zoom_level = scale


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
# Auto Layout — organiza nós por nível de dependência
# ---------------------------------------------------------------------------

def _compute_graph() -> tuple[dict, dict, dict]:
    """Retorna (levels, successors, predecessors) para todos os glyphs."""
    glyph_ids = list(APP_STATE["glyphs"].keys())
    successors:   dict[str, list[str]] = {gid: [] for gid in glyph_ids}
    predecessors: dict[str, list[str]] = {gid: [] for gid in glyph_ids}
    in_degree:    dict[str, int]       = {gid: 0  for gid in glyph_ids}

    for _link_tag, (out_attr, in_attr) in APP_STATE["links"].items():
        out_info = _attr_tag_to_port.get(out_attr)
        in_info  = _attr_tag_to_port.get(in_attr)
        if not out_info or not in_info:
            continue
        out_gid, in_gid = out_info[0], in_info[0]
        if out_gid != in_gid and out_gid in successors and in_gid in successors:
            if in_gid not in successors[out_gid]:
                successors[out_gid].append(in_gid)
                predecessors[in_gid].append(out_gid)
                in_degree[in_gid] += 1

    # Kahn's + longest-path
    queue  = [gid for gid in glyph_ids if in_degree[gid] == 0]
    levels: dict[str, int] = {gid: 0 for gid in queue}

    while queue:
        node = queue.pop(0)
        for succ in successors[node]:
            lvl = levels[node] + 1
            if lvl > levels.get(succ, 0):
                levels[succ] = lvl
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    for gid in glyph_ids:
        if gid not in levels:
            levels[gid] = 0

    return levels, successors, predecessors


def auto_layout(direction: str = "LR"):
    """
    Reorganiza os nós no canvas.
    direction="LR" → esquerda para a direita (colunas) — compact layout:
        • vglCreateImage com 1 sucessor fica na mesma coluna do consumidor (acima)
        • nós terminais (SaveImage, ShowImage) ficam na mesma coluna do predecessor (abaixo)
    direction="TB" → cima para baixo (linhas)
    """
    if not APP_STATE["glyphs"]:
        return

    levels, successors, predecessors = _compute_graph()

    if direction == "TB":
        level_groups: dict[int, list[str]] = {}
        for gid, lvl in levels.items():
            level_groups.setdefault(lvl, []).append(gid)

        COLS        = 3
        NODE_X      = 300
        NODE_Y      = 200
        X_OFFSET, Y_OFFSET = 60, 60

        current_row = 0
        for lvl, gids in sorted(level_groups.items()):
            gids.sort()
            sub_rows = (len(gids) + COLS - 1) // COLS
            for i, gid in enumerate(gids):
                col     = i % COLS
                sub_row = i // COLS
                x = X_OFFSET + col * NODE_X
                y = Y_OFFSET + (current_row + sub_row) * NODE_Y
                state = APP_STATE["glyphs"][gid]
                if dpg.does_item_exist(state.dpg_node_tag):
                    dpg.set_item_pos(state.dpg_node_tag, [x, y])
            current_row += sub_rows + (1 if lvl < max(level_groups) else 0)

    else:
        # ── Compact LR ───────────────────────────────────────────────────────
        LEVEL_SPACING = 280   # distância horizontal entre colunas
        NODE_SPACING  = 200   # distância vertical entre nós principais
        SLOT_H        = 110   # altura de slot para nós auxiliares
        X_OFFSET      = 60
        Y_MAIN        = 130   # y base dos nós principais (deixa espaço para buffer acima)

        # Detecta "buffer" nodes: vglCreateImage com exatamente 1 sucessor
        buffer_parent: dict[str, str] = {}
        for gid, state in APP_STATE["glyphs"].items():
            succs = successors.get(gid, [])
            if state.func == "vglCreateImage" and len(succs) == 1:
                consumer = succs[0]
                buffer_parent[gid] = consumer
                levels[gid] = levels[consumer]   # coloca na mesma coluna

        # Detecta nós terminais: sem sucessores, com exatamente 1 predecessor
        terminal_parent: dict[str, str] = {}
        for gid in list(APP_STATE["glyphs"].keys()):
            if not successors.get(gid) and len(predecessors.get(gid, [])) == 1:
                pred = predecessors[gid][0]
                terminal_parent[gid] = pred
                levels[gid] = levels[pred]       # coloca na mesma coluna

        # Reconstrói grupos com os níveis ajustados
        level_groups = {}
        for gid, lvl in levels.items():
            level_groups.setdefault(lvl, []).append(gid)

        # Mapeia nível → índice de coluna (sem buracos)
        sorted_lvls = sorted(level_groups.keys())

        node_y: dict[str, int] = {}   # guarda y final de cada nó

        for col_idx, lvl in enumerate(sorted_lvls):
            x = X_OFFSET + col_idx * LEVEL_SPACING
            gids = level_groups[lvl]

            buffers   = [g for g in gids if g in buffer_parent]
            terminals = [g for g in gids if g in terminal_parent]
            mains     = sorted(g for g in gids
                               if g not in buffer_parent and g not in terminal_parent)

            # 1) Nós principais
            for row, gid in enumerate(mains):
                y = Y_MAIN + row * NODE_SPACING
                node_y[gid] = y
                state = APP_STATE["glyphs"][gid]
                if dpg.does_item_exist(state.dpg_node_tag):
                    dpg.set_item_pos(state.dpg_node_tag, [x, y])

            # 2) Buffers acima do seu consumidor
            consumer_slot: dict[str, int] = {}
            for gid in buffers:
                consumer = buffer_parent[gid]
                cy = node_y.get(consumer, Y_MAIN)
                slot = consumer_slot.get(consumer, 0)
                y = cy - SLOT_H * (slot + 1)
                consumer_slot[consumer] = slot + 1
                node_y[gid] = y
                state = APP_STATE["glyphs"][gid]
                if dpg.does_item_exist(state.dpg_node_tag):
                    dpg.set_item_pos(state.dpg_node_tag, [x, y])

            # 3) Terminais abaixo do seu predecessor
            pred_slot: dict[str, int] = {}
            for gid in sorted(terminals):
                pred = terminal_parent[gid]
                py = node_y.get(pred, Y_MAIN)
                slot = pred_slot.get(pred, 0)
                y = py + NODE_SPACING + slot * SLOT_H
                pred_slot[pred] = slot + 1
                node_y[gid] = y
                state = APP_STATE["glyphs"][gid]
                if dpg.does_item_exist(state.dpg_node_tag):
                    dpg.set_item_pos(state.dpg_node_tag, [x, y])

    APP_STATE["dirty"] = True


# ---------------------------------------------------------------------------
# Expor mapeamento para wksp_io
# ---------------------------------------------------------------------------

def get_attr_tag_to_port() -> dict[int, tuple[str, str]]:
    return _attr_tag_to_port


def get_zoom_level() -> float:
    return _zoom_level


# ---------------------------------------------------------------------------
# Copy / Paste / Duplicate
# ---------------------------------------------------------------------------

_copy_buffer: list[dict] = []  # [{"func": str, "params": dict, "dx": float, "dy": float}]


def copy_selected():
    """Copia nós selecionados para o buffer interno."""
    selected = dpg.get_selected_nodes(NODE_EDITOR_TAG)
    if not selected:
        return
    tag_to_gid = {s.dpg_node_tag: gid for gid, s in APP_STATE["glyphs"].items()}
    entries = []
    positions = []
    for node_tag in selected:
        gid = tag_to_gid.get(node_tag)
        if not gid:
            continue
        state = APP_STATE["glyphs"][gid]
        pos = dpg.get_item_pos(node_tag)
        positions.append(pos)
        entries.append({"func": state.func, "params": dict(state.params), "pos": pos})
    if not entries:
        return
    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    _copy_buffer.clear()
    for e in entries:
        _copy_buffer.append({
            "func":   e["func"],
            "params": e["params"],
            "dx":     e["pos"][0] - min_x,
            "dy":     e["pos"][1] - min_y,
        })


def paste_nodes(base_x: float = None, base_y: float = None):
    """Cola nós do buffer com offset."""
    if not _copy_buffer:
        return
    if base_x is None:
        try:
            sz = dpg.get_item_rect_size("canvas_win")
            base_x, base_y = sz[0] / 2 + 40, sz[1] / 2 + 40
        except Exception:
            base_x, base_y = 200.0, 200.0
    from gui.history import push_undo
    push_undo()
    for entry in _copy_buffer:
        add_glyph_to_canvas(
            entry["func"],
            pos_x=base_x + entry["dx"] + 40,
            pos_y=base_y + entry["dy"] + 40,
            params=entry["params"],
        )


def duplicate_selected():
    """Duplica os nós selecionados com offset de 40px."""
    selected = dpg.get_selected_nodes(NODE_EDITOR_TAG)
    if not selected:
        return
    copy_selected()
    if not _copy_buffer:
        return
    try:
        pos = dpg.get_item_pos(selected[0])
        paste_nodes(base_x=pos[0], base_y=pos[1])
    except Exception:
        paste_nodes()


# ---------------------------------------------------------------------------
# Context menu (right-click)
# ---------------------------------------------------------------------------

def _show_node_context_menu(gid: str):
    """Popup de contexto para um nó específico."""
    mx, my = dpg.get_mouse_pos(local=False)
    popup_tag = _new_tag()
    state = APP_STATE["glyphs"].get(gid)
    with dpg.window(tag=popup_tag, pos=[int(mx), int(my)],
                    no_title_bar=True, no_move=True, no_resize=True,
                    no_scrollbar=True, popup=True, min_size=[140, 10]):
        if state:
            dpg.add_text(state.func, color=(180, 210, 255, 255))
            dpg.add_separator()
        dpg.add_menu_item(
            label=t("ctx_info"),
            callback=lambda: (dpg.delete_item(popup_tag), show_info(gid)),
        )
        dpg.add_menu_item(
            label=t("ctx_duplicate"),
            callback=lambda: (dpg.delete_item(popup_tag), _dup_single(gid)),
        )
        dpg.add_separator()
        dpg.add_menu_item(
            label=t("ctx_delete"),
            callback=lambda: (dpg.delete_item(popup_tag), remove_glyph(gid)),
        )


def _dup_single(gid: str):
    """Duplica um único nó com offset."""
    state = APP_STATE["glyphs"].get(gid)
    if not state or not dpg.does_item_exist(state.dpg_node_tag):
        return
    pos = dpg.get_item_pos(state.dpg_node_tag)
    from gui.history import push_undo
    push_undo()
    add_glyph_to_canvas(state.func,
                        pos_x=pos[0] + 40,
                        pos_y=pos[1] + 40,
                        params=dict(state.params))


def _show_canvas_context_menu(drop_x: float, drop_y: float):
    """Popup de contexto para área vazia do canvas — categorias de funções."""
    mx, my = dpg.get_mouse_pos(local=False)
    popup_tag = _new_tag()
    with dpg.window(tag=popup_tag, pos=[int(mx), int(my)],
                    no_title_bar=True, no_move=True, no_resize=True,
                    no_scrollbar=True, popup=True, min_size=[180, 10]):
        dpg.add_text(t("ctx_add_node"), color=(180, 220, 180, 255))
        dpg.add_separator()
        from gui.glyph_registry import GLYPH_REGISTRY, CATEGORIES
        for category in CATEGORIES:
            if category == "Procedures":
                continue
            funcs = [g for g in GLYPH_REGISTRY.values() if g.category == category]
            if not funcs:
                continue
            with dpg.menu(label=category):
                for glyph_def in funcs:
                    def _cb(f=glyph_def.func, px=drop_x, py=drop_y):
                        dpg.delete_item(popup_tag)
                        add_glyph_to_canvas(f, pos_x=px, pos_y=py)
                    dpg.add_menu_item(label=glyph_def.label, callback=_cb)


def _on_right_click(sender, app_data):
    """Handler de clique direito global."""
    if not dpg.does_item_exist("canvas_win"):
        return
    if not dpg.is_item_hovered("canvas_win"):
        return
    # Verifica se algum nó está sob o cursor
    for gid, state in list(APP_STATE["glyphs"].items()):
        if dpg.does_item_exist(state.dpg_node_tag) and dpg.is_item_hovered(state.dpg_node_tag):
            _show_node_context_menu(gid)
            return
    # Canvas vazio — calcula posição local
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
    _show_canvas_context_menu(cx, cy)


# ---------------------------------------------------------------------------
# Preview de imagem nos nós
# ---------------------------------------------------------------------------

_PREVIEW_MAX_PX = 200   # lado maior do thumbnail


def queue_node_preview(glyph_id: str, image_path: str):
    """Thread-safe: enfileira um preview de imagem para um nó."""
    with _preview_lock:
        _preview_queue.append((glyph_id, image_path))


def flush_node_previews():
    """Chamado no render loop da thread principal."""
    if not _preview_queue:
        return
    with _preview_lock:
        items = _preview_queue.copy()
        _preview_queue.clear()

    from gui.image_preview import get_texture_registry_tag
    tex_reg = get_texture_registry_tag()

    for glyph_id, image_path in items:
        state = APP_STATE["glyphs"].get(glyph_id)
        if not state or not dpg.does_item_exist(state.dpg_node_tag):
            continue
        _apply_node_preview(state, image_path, tex_reg)


def _apply_node_preview(state, image_path: str, tex_reg: int):
    """Cria/atualiza o thumbnail dentro do nó (main thread only)."""
    from PIL import Image
    import numpy as np

    # Remove preview anterior
    if state.preview_attr_tag and dpg.does_item_exist(state.preview_attr_tag):
        dpg.delete_item(state.preview_attr_tag)
        state.preview_attr_tag = 0
    if state.preview_tex_tag and dpg.does_item_exist(state.preview_tex_tag):
        dpg.delete_item(state.preview_tex_tag)
        state.preview_tex_tag = 0

    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"[node_preview] Erro ao abrir {image_path}: {e}")
        return

    # Redimensiona mantendo aspecto
    w, h = img.size
    scale = min(_PREVIEW_MAX_PX / max(w, h), 1.0)
    tw = max(int(w * scale), 1)
    th = max(int(h * scale), 1)
    img = img.resize((tw, th), Image.LANCZOS)

    data = np.array(img, dtype=np.float32) / 255.0

    tex_tag = dpg.generate_uuid()
    dpg.add_static_texture(
        width=tw, height=th,
        default_value=data.flatten().tolist(),
        tag=tex_tag,
        parent=tex_reg,
    )

    attr_tag = dpg.generate_uuid()
    with dpg.node_attribute(
        tag=attr_tag,
        parent=state.dpg_node_tag,
        attribute_type=dpg.mvNode_Attr_Static,
    ):
        dpg.add_image(tex_tag, width=tw, height=th)

    state.preview_attr_tag = attr_tag
    state.preview_tex_tag  = tex_tag
