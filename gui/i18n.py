"""
Internationalisation for VGLGui.
Usage:
    from gui.i18n import t, set_language, get_language
    label = t("menu_file")   # returns string in current language
"""

_LANG = "pt"   # runtime-mutable; set by set_language() before any UI is built

_STRINGS: dict[str, dict[str, str]] = {
    # ── Geral ────────────────────────────────────────────────────────────────
    "untitled":     {"pt": "Sem título",        "en": "Untitled"},
    "confirm":      {"pt": "Confirmar",          "en": "Confirm"},
    "yes":          {"pt": "Sim",                "en": "Yes"},
    "no":           {"pt": "Não",               "en": "No"},
    "close":        {"pt": "Fechar",             "en": "Close"},

    # ── Menu File ────────────────────────────────────────────────────────────
    "menu_file":    {"pt": "Arquivo",            "en": "File"},
    "new":          {"pt": "Novo           Ctrl+N",  "en": "New           Ctrl+N"},
    "open":         {"pt": "Abrir...       Ctrl+O",  "en": "Open...       Ctrl+O"},
    "save":         {"pt": "Salvar         Ctrl+S",  "en": "Save          Ctrl+S"},
    "save_as":      {"pt": "Salvar como    Ctrl+Shift+S", "en": "Save As       Ctrl+Shift+S"},
    "exit":         {"pt": "Sair",               "en": "Exit"},

    # ── Menu Run ─────────────────────────────────────────────────────────────
    "menu_run":     {"pt": "Executar",           "en": "Run"},
    "run_gpu":      {"pt": "Executar GPU   F5",  "en": "Run GPU       F5"},
    "run_cpu":      {"pt": "Executar CPU",       "en": "Run CPU"},
    "stop":         {"pt": "Parar          F6",  "en": "Stop          F6"},

    # ── Menu Examples ────────────────────────────────────────────────────────
    "menu_examples":{"pt": "Exemplos",           "en": "Examples"},
    "not_found":    {"pt": "(não encontrado)",    "en": "(not found)"},

    # ── Menu View ────────────────────────────────────────────────────────────
    "menu_view":    {"pt": "Exibir",             "en": "View"},
    "layout_lr":    {"pt": "Layout: E→D   Ctrl+L",         "en": "Layout: L→R  Ctrl+L"},
    "layout_tb":    {"pt": "Layout: C→B",                  "en": "Layout: T→B"},
    "fit_screen":   {"pt": "Ajustar à Tela  Ctrl+Shift+F", "en": "Fit to Screen  Ctrl+Shift+F"},
    "toggle_log":        {"pt": "Alternar Log",                  "en": "Toggle Log"},
    "toggle_cat_colors": {"pt": "Alternar Cores por Categoria",  "en": "Toggle Category Colors"},

    # ── Menu Help ────────────────────────────────────────────────────────────
    "menu_help":        {"pt": "Ajuda",                    "en": "Help"},
    "help_shortcuts":   {"pt": "Atalhos de Teclado",       "en": "Keyboard Shortcuts"},
    "help_about":       {"pt": "Sobre",                    "en": "About"},
    "undo":             {"pt": "Desfazer",                 "en": "Undo"},
    "redo":             {"pt": "Refazer",                  "en": "Redo"},
    "delete_selected":  {"pt": "Deletar selecionado",      "en": "Delete selected"},
    "zoom_hint":        {"pt": "Zoom in/out no canvas",    "en": "Zoom in/out canvas"},
    "run_stop":         {"pt": "Parar execução",           "en": "Stop execution"},
    "about_desc":       {"pt": "Editor visual de workflows de processamento de imagem.",
                         "en": "Visual workflow editor for image processing."},
    "about_framework":  {"pt": "Interface: Dear PyGui",    "en": "GUI: Dear PyGui"},
    "about_backend":    {"pt": "Backend: VisionGL (OpenCL)","en": "Backend: VisionGL (OpenCL)"},

    # ── Menu Language ────────────────────────────────────────────────────────
    "menu_language":{"pt": "Idioma",             "en": "Language"},
    "lang_pt":      {"pt": "Português (PT-BR)",  "en": "Português (PT-BR)"},
    "lang_en":      {"pt": "English",            "en": "English"},

    # ── Dialogs / Confirmações ───────────────────────────────────────────────
    "unsaved_load": {"pt": "Há alterações não salvas. Carregar o exemplo mesmo assim?",
                     "en": "There are unsaved changes. Load example anyway?"},
    "unsaved_new":  {"pt": "Há alterações não salvas. Continuar?",
                     "en": "There are unsaved changes. Continue?"},

    # ── File dialogs ─────────────────────────────────────────────────────────
    "open_workflow":{"pt": "Abrir fluxo de trabalho", "en": "Open Workflow"},
    "save_workflow":{"pt": "Salvar fluxo de trabalho","en": "Save Workflow"},
    "select_file":  {"pt": "Selecionar arquivo",  "en": "Select file"},
    "export_canvas":{"pt": "Exportar canvas como imagem...", "en": "Export Canvas as Image..."},
    "export_canvas_saved":  {"pt": "[export] Canvas exportado: {path}", "en": "[export] Canvas exported: {path}"},
    "export_canvas_error":  {"pt": "[export] Erro ao exportar canvas: {err}", "en": "[export] Error exporting canvas: {err}"},
    "export_canvas_no_tool":{"pt": "[export] Nenhuma ferramenta de captura disponível (mss, scrot ou ImageMagick).",
                              "en": "[export] No screenshot tool available (mss, scrot, or ImageMagick)."},
    "export_canvas_dialog": {"pt": "Exportar Canvas — Salvar PNG", "en": "Export Canvas — Save PNG"},

    # ── Log panel ────────────────────────────────────────────────────────────
    "exec_log":     {"pt": "Log de execução",    "en": "Execution Log"},
    "clear":        {"pt": "Limpar",             "en": "Clear"},
    "copy":         {"pt": "Copiar",             "en": "Copy"},

    # ── Sidebar ──────────────────────────────────────────────────────────────
    "vgl_functions":{"pt": "Funções VGL",        "en": "VGL Functions"},
    "search_hint":  {"pt": "Buscar...",           "en": "Search..."},

    # ── Strel widget ─────────────────────────────────────────────────────────
    "generate_se":  {"pt": "Gerar SE",           "en": "Generate SE"},
    "active_px":    {"pt": "ok: {ones}/{total} px ativos",
                     "en": "ok: {ones}/{total} active px"},

    # ── Info popup ───────────────────────────────────────────────────────────
    "info_title":   {"pt": "Info - {func}",      "en": "Info - {func}"},
    "func_label":   {"pt": "Função:    {func}",  "en": "Function:  {func}"},
    "lib_label":    {"pt": "Biblioteca: {lib}",  "en": "Library:   {lib}"},
    "cat_label":    {"pt": "Categoria:  {cat}",  "en": "Category:  {cat}"},
    "ports_label":  {"pt": "Portas:",            "en": "Ports:"},
    "params_label": {"pt": "Parâmetros:",        "en": "Parameters:"},
    "optional":     {"pt": " (opcional)",        "en": " (optional)"},

    # ── Context menu ─────────────────────────────────────────────────────────
    "ctx_info":      {"pt": "Info",           "en": "Info"},
    "ctx_duplicate": {"pt": "Duplicar",       "en": "Duplicate"},
    "ctx_delete":    {"pt": "Deletar",        "en": "Delete"},
    "ctx_add_node":  {"pt": "Adicionar nó",   "en": "Add Node"},

    # ── Runner ───────────────────────────────────────────────────────────────
    "runner_running":  {"pt": "[runner] Execução já em andamento.",
                        "en": "[runner] Execution already in progress."},
    "runner_empty":    {"pt": "[runner] Canvas vazio — nada para executar.",
                        "en": "[runner] Canvas is empty — nothing to execute."},
    "runner_starting": {"pt": "\n[runner] Iniciando execução em {device}...\n",
                        "en": "\n[runner] Starting execution on {device}...\n"},
    "runner_error":    {"pt": "[runner] ERRO ao iniciar processo: {err}",
                        "en": "[runner] ERROR starting process: {err}"},
    "runner_stopped":  {"pt": "[runner] Execução interrompida pelo usuário.",
                        "en": "[runner] Execution stopped by user."},
    "runner_finished": {"pt": "\n[runner] Execução concluída: {label}\n",
                        "en": "\n[runner] Execution finished: {label}\n"},
    "runner_ok":       {"pt": "OK",              "en": "OK"},
    "runner_err_code": {"pt": "ERRO (código {code})", "en": "ERROR (code {code})"},
    "runner_time":     {"pt": "[runner] Tempo total: {secs}s", "en": "[runner] Total time: {secs}s"},

    # ── wksp_io ──────────────────────────────────────────────────────────────
    "wksp_not_found":   {"pt": "[wksp_io] Arquivo não encontrado: {path}",
                         "en": "[wksp_io] File not found: {path}"},
    "wksp_error_read":  {"pt": "[wksp_io] Erro ao ler {path}: {err}",
                         "en": "[wksp_io] Error reading {path}: {err}"},
    "wksp_unknown_func":{"pt": "[wksp_io] Função desconhecida: '{func}'",
                         "en": "[wksp_io] Unknown function: '{func}'"},
    "wksp_error_glyph": {"pt": "[wksp_io] Erro ao criar glyph '{func}': {err}",
                         "en": "[wksp_io] Error creating glyph '{func}': {err}"},
    "wksp_loaded":      {"pt": "[wksp_io] Carregado: {name}  ({ok}/{total} glyphs, {links} conexões)",
                         "en": "[wksp_io] Loaded: {name}  ({ok}/{total} glyphs, {links} connections)"},
    "wksp_layout_error":{"pt": "[wksp_io] Erro no auto_layout:\n{tb}",
                         "en": "[wksp_io] Error in auto_layout:\n{tb}"},

    # ── Procedures ───────────────────────────────────────────────────────────
    "proc_new":         {"pt": "Nova Procedure",       "en": "New Procedure"},
    "proc_open":        {"pt": "Abrir",                "en": "Open"},
    "proc_close_popup": {"pt": "Fechar",               "en": "Close"},
    "proc_editor":      {"pt": "Procedure: {name}",    "en": "Procedure: {name}"},
    "proc_name_label":  {"pt": "Nome da procedure:",   "en": "Procedure name:"},
    "proc_name_hint":   {"pt": "Ex: Demo",             "en": "Ex: Demo"},
    "proc_create":      {"pt": "Criar",                "en": "Create"},
    "proc_cancel":      {"pt": "Cancelar",             "en": "Cancel"},
    "proc_name_empty":  {"pt": "O nome não pode ser vazio.", "en": "Name cannot be empty."},
    "proc_name_dup":    {"pt": "Já existe uma procedure com esse nome.", "en": "A procedure with this name already exists."},
    "ext_input":        {"pt": "Entrada Externa",      "en": "External Input"},
    "ext_output":       {"pt": "Saída Externa",        "en": "External Output"},
    "cat_procedures":   {"pt": "Procedures",           "en": "Procedures"},

    # ── Category display names ───────────────────────────────────────────────
    "cat_io":         {"pt": "E/S",              "en": "I/O"},
    "cat_allocation": {"pt": "Alocação",         "en": "Allocation"},
    "cat_color":      {"pt": "Cor",              "en": "Color"},
    "cat_filters":    {"pt": "Filtros",          "en": "Filters"},
    "cat_morphology": {"pt": "Morfologia",       "en": "Morphology"},
    "cat_operations": {"pt": "Operações",        "en": "Operations"},
    "cat_special":    {"pt": "Especial",         "en": "Special"},
    "cat_nd":         {"pt": "ND",               "en": "ND"},
    "cat_3d":         {"pt": "3D",               "en": "3D"},
    "cat_fuzzy2d":    {"pt": "Fuzzy 2D",         "en": "Fuzzy 2D"},
    "cat_fuzzy3d":    {"pt": "Fuzzy 3D",         "en": "Fuzzy 3D"},

    # ── Status bar ───────────────────────────────────────────────────────────────
    "status_nodes":  {"pt": "Nós",     "en": "Nodes"},
    "status_links":  {"pt": "Links",   "en": "Links"},
    "status_zoom":   {"pt": "Zoom",    "en": "Zoom"},

    # ── Recent files ─────────────────────────────────────────────────────────────
    "recent_files":  {"pt": "Recentes",          "en": "Recent Files"},
    "recent_empty":  {"pt": "(vazio)",            "en": "(empty)"},
}

# Internal category key → i18n key
_CATEGORY_KEY: dict[str, str] = {
    "Procedures":  "cat_procedures",
    "I/O":         "cat_io",
    "Allocation":  "cat_allocation",
    "Color":       "cat_color",
    "Filters":     "cat_filters",
    "Morphology":  "cat_morphology",
    "Operations":  "cat_operations",
    "Special":     "cat_special",
    "ND":          "cat_nd",
    "3D":          "cat_3d",
    "Fuzzy 2D":    "cat_fuzzy2d",
    "Fuzzy 3D":    "cat_fuzzy3d",
}


def get_language() -> str:
    """Returns the current language code ('pt' or 'en')."""
    return _LANG


def set_language(lang: str):
    """Sets the current language. Must be called before the UI is built."""
    global _LANG
    if lang in ("pt", "en"):
        _LANG = lang


def t(key: str, **kwargs) -> str:
    """
    Returns the translation for *key* in the current language.
    Supports str.format-style placeholders: t("runner_starting", device="GPU").
    Falls back to the key itself if translation is missing.
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(_LANG) or entry.get("en") or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    return text


def cat_label(internal_name: str) -> str:
    """
    Returns the translated display name for a category internal key.
    E.g. cat_label("I/O") → "E/S" in PT-BR.
    """
    key = _CATEGORY_KEY.get(internal_name)
    if key:
        return t(key)
    return internal_name
