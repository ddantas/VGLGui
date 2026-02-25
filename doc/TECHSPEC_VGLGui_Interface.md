# Tech Spec — VGLGui Interface
**Versão:** 1.0
**Data:** 2026-02-25
**Referência:** PRD `doc/PRD_VGLGui_Interface.md`

---

## 1. Visão Geral da Arquitetura

```
┌──────────────────────────────────────────────────────────────┐
│                    gui/main.py  (entry point)                │
│                                                              │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │ sidebar  │  │              app.py                      │ │
│  │ _glyphs  │  │  ┌─────────────────────────────────────┐ │ │
│  │ .py      │  │  │  canvas.py  (dpg.node_editor)       │ │ │
│  │          │  │  │  - GlyphNode items                  │ │ │
│  │ categoria│  │  │  - Edge links                       │ │ │
│  │ + lista  │  │  │  - pan/zoom                         │ │ │
│  │ de funcs │  │  └─────────────────────────────────────┘ │ │
│  └──────────┘  │  ┌──────────┐  ┌────────────────────────┐ │ │
│                │  │toolbar.py│  │    log_panel.py         │ │ │
│                │  └──────────┘  └────────────────────────┘ │ │
│                └──────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │ glyph_registry.py│  │         wksp_io.py               │ │
│  │  (catálogo VGL)  │  │  (leitura/escrita .wksp)         │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
         │ subprocess.Popen                │ from readWorkflow import
         ▼                                 ▼
   execWorkflow.py                   readWorkflow.py
   (executor OpenCL)                 (parser .wksp)
```

---

## 2. Estrutura de Arquivos

```
VGLGui/
├── gui/
│   ├── __init__.py
│   ├── main.py              # entry point: cria contexto dpg, chama app.run()
│   ├── app.py               # layout principal, tema, callbacks globais
│   ├── canvas.py            # node editor: Glyphs, conexões, interação
│   ├── sidebar_glyphs.py    # painel esquerdo: paleta de funções
│   ├── toolbar.py           # menu File/Run/View e atalhos de teclado
│   ├── log_panel.py         # painel inferior de log em tempo real
│   ├── glyph_registry.py    # definição estática de todos os Glyphs VGL
│   └── wksp_io.py           # serializa/deserializa .wksp ↔ estado GUI
├── readWorkflow.py           # (existente, sem modificação)
└── execWorkflow.py           # (existente, sem modificação na v1)
```

---

## 3. Estado Global da Aplicação (`app.py`)

Um único dicionário de estado mantido em memória durante a sessão:

```python
# gui/app.py
APP_STATE = {
    # Workspace atual
    "wksp_path": None,          # str | None — caminho do arquivo aberto
    "dirty": False,             # bool — há mudanças não salvas?
    "device": "GPU",            # "GPU" | "CPU"

    # Glyphs no canvas: glyph_id → GlyphState
    "glyphs": {},               # dict[str, GlyphState]

    # Conexões ativas: link_id → (output_attr_tag, input_attr_tag)
    "links": {},                # dict[int, tuple[int, int]]

    # Execução
    "exec_process": None,       # subprocess.Popen | None
    "exec_running": False,      # bool

    # Próximo ID único para Glyphs novos
    "next_glyph_id": 1,
}
```

### `GlyphState` (dataclass)

```python
# gui/app.py
from dataclasses import dataclass, field

@dataclass
class GlyphState:
    glyph_id: str                    # ex: "3"
    func: str                        # ex: "vglClConvolution"
    library: str                     # ex: "VGL_CL"
    pos_x: float                     # posição no canvas
    pos_y: float
    params: dict[str, str]           # nome → valor (string)
    status: str = "ready"            # "ready" | "running" | "done" | "error"

    # tags Dear PyGui para lookup
    dpg_node_tag: int = 0
    dpg_attr_tags: dict = field(default_factory=dict)  # port_name → dpg_tag
```

---

## 4. `glyph_registry.py` — Catálogo de Funções VGL

Define metadados de todos os Glyphs suportados. Nenhuma lógica de execução aqui.

```python
# gui/glyph_registry.py
from dataclasses import dataclass, field

@dataclass
class PortDef:
    name: str
    kind: str   # "input" | "output"
    required: bool = True

@dataclass
class ParamDef:
    name: str
    type: str           # "text" | "int" | "float" | "array" | "file" | "bool"
    default: str = ""
    label: str = ""     # display name (se diferente do name)

@dataclass
class GlyphDef:
    func: str           # nome exato como aparece no .wksp
    label: str          # nome de exibição (pode ser igual ao func)
    category: str
    ports: list[PortDef]
    params: list[ParamDef]
    library: str = "VGL_CL"

GLYPH_REGISTRY: dict[str, GlyphDef] = {
    # --- I/O ---
    "vglLoad2dImage": GlyphDef(
        func="vglLoad2dImage", label="vglLoadImage", category="I/O",
        ports=[PortDef("RETVAL", "output")],
        params=[
            ParamDef("filename", "file", label="filename"),
            ParamDef("iscolor",  "bool", default="1"),
            ParamDef("has_mipmap", "bool", default="0"),
        ]
    ),
    "vglSaveImage": GlyphDef(
        func="vglSaveImage", label="vglSaveImage", category="I/O",
        ports=[PortDef("image", "input")],
        params=[ParamDef("filename", "file", label="filename")]
    ),
    "ShowImage": GlyphDef(
        func="ShowImage", label="ShowImage", category="I/O",
        ports=[PortDef("image", "input")],
        params=[]
    ),
    # --- Alocação ---
    "vglCreateImage": GlyphDef(
        func="vglCreateImage", label="vglCreateImage", category="Alocação",
        ports=[PortDef("img", "input"), PortDef("RETVAL", "output")],
        params=[]
    ),
    # --- Cor ---
    "vglClRgb2Gray": GlyphDef(
        func="vglClRgb2Gray", label="vglClRgb2Gray", category="Cor",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    "vglClSwapRgb": GlyphDef(
        func="vglClSwapRgb", label="vglClSwapRgb", category="Cor",
        ports=[PortDef("src","input"), PortDef("dst","input"), PortDef("dst","output")],
        params=[]
    ),
    # --- Filtros ---
    "vglClConvolution": GlyphDef(
        func="vglClConvolution", label="vglClConvolution", category="Filtros",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[
            ParamDef("convolution_window", "array", default="[0,0,0.125,0,0.125,0.5,0.125,0,0.125,0,0]"),
            ParamDef("window_size_x", "int", default="3"),
            ParamDef("window_size_y", "int", default="3"),
        ]
    ),
    "vglClBlurSq3": GlyphDef(
        func="vglClBlurSq3", label="vglClBlurSq3", category="Filtros",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    # --- Morfologia ---
    "vglClDilate": GlyphDef(
        func="vglClDilate", label="vglClDilate", category="Morfologia",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[
            ParamDef("convolution_window", "array", default="[1,1,1,1,1,1,1,1,1]"),
            ParamDef("window_size_x", "int", default="3"),
            ParamDef("window_size_y", "int", default="3"),
        ]
    ),
    "vglClErode": GlyphDef(
        func="vglClErode", label="vglClErode", category="Morfologia",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[
            ParamDef("convolution_window", "array", default="[1,1,1,1,1,1,1,1,1]"),
            ParamDef("window_size_x", "int", default="3"),
            ParamDef("window_size_y", "int", default="3"),
        ]
    ),
    # --- Operações ---
    "vglClSub": GlyphDef(
        func="vglClSub", label="vglClSub", category="Operações",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    "vglClSum": GlyphDef(
        func="vglClSum", label="vglClSum", category="Operações",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    "vglClMax": GlyphDef(
        func="vglClMax", label="vglClMax", category="Operações",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    "vglClMin": GlyphDef(
        func="vglClMin", label="vglClMin", category="Operações",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    "vglClThreshold": GlyphDef(
        func="vglClThreshold", label="vglClThreshold", category="Operações",
        ports=[PortDef("src","input"), PortDef("dst","input"), PortDef("dst","output")],
        params=[ParamDef("thresh", "float", default="0.5")]
    ),
    "vglClInvert": GlyphDef(
        func="vglClInvert", label="vglClInvert", category="Operações",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[]
    ),
    # --- Especiais ---
    "Reconstruct": GlyphDef(
        func="Reconstruct", label="Reconstruct", category="Especiais",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"), PortDef("img_output","output")],
        params=[
            ParamDef("convolution_window", "array"),
            ParamDef("window_size_x", "int", default="3"),
            ParamDef("window_size_y", "int", default="3"),
        ]
    ),
    # ... (demais funções seguem o mesmo padrão)
}

CATEGORIES = ["I/O", "Alocação", "Cor", "Filtros", "Morfologia", "Operações", "Especiais", "ND", "3D"]
```

> **Nota:** O registry completo deve cobrir todas as ~67 funções encontradas em `execWorkflow.py`. A v1 foca nas funções dos workflows de exemplo (`demo.wksp`, `fundus.wksp`).

---

## 5. `canvas.py` — Node Editor

### 5.1 Tags Dear PyGui

Dear PyGui usa **inteiros** como tags (IDs únicos). Gerenciar com um contador global:

```python
# gui/canvas.py
import dearpygui.dearpygui as dpg
from gui.app import APP_STATE, GlyphState
from gui.glyph_registry import GLYPH_REGISTRY

_tag_counter = 1000  # começa depois de tags reservadas

def _new_tag() -> int:
    global _tag_counter
    _tag_counter += 1
    return _tag_counter

# Tag do node_editor — referência global
NODE_EDITOR_TAG = 100
```

### 5.2 Criação de Glyph no Canvas

```python
def add_glyph_to_canvas(func: str, pos_x: float = 100, pos_y: float = 100) -> str:
    """
    Adiciona um Glyph ao canvas e ao APP_STATE.
    Retorna o glyph_id gerado.
    """
    glyph_def = GLYPH_REGISTRY[func]
    glyph_id = str(APP_STATE["next_glyph_id"])
    APP_STATE["next_glyph_id"] += 1

    node_tag = _new_tag()
    attr_tags = {}

    with dpg.node(label=glyph_def.label, parent=NODE_EDITOR_TAG,
                  pos=[pos_x, pos_y], tag=node_tag):

        # Botões de controle (Delete, Expand, Info) na barra de título
        # Dear PyGui não suporta botões no título nativamente —
        # adicionar como primeiro node_attribute estático
        ctrl_tag = _new_tag()
        with dpg.node_attribute(label="##ctrl", tag=ctrl_tag,
                                attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_button(label="X", width=20,
                           callback=lambda: remove_glyph(glyph_id))
            dpg.add_same_line()
            dpg.add_button(label="≡", width=20,
                           callback=lambda: toggle_expand(glyph_id))
            dpg.add_same_line()
            dpg.add_button(label="i", width=20,
                           callback=lambda: show_info(glyph_id))
            dpg.add_same_line()
            dpg.add_text(f"[{glyph_id}]")

        # Portas de entrada
        for port in glyph_def.ports:
            if port.kind == "input":
                tag = _new_tag()
                attr_tags[port.name + "_in"] = tag
                with dpg.node_attribute(label=port.name, tag=tag,
                                        attribute_type=dpg.mvNode_Attr_Input):
                    dpg.add_text(port.name)

        # Parâmetros (modo compacto: ocultos por padrão)
        for param in glyph_def.params:
            tag = _new_tag()
            attr_tags["param_" + param.name] = tag
            with dpg.node_attribute(label=f"##{param.name}",
                                    tag=tag,
                                    attribute_type=dpg.mvNode_Attr_Static):
                _add_param_widget(param, glyph_id)

        # Portas de saída
        for port in glyph_def.ports:
            if port.kind == "output":
                tag = _new_tag()
                attr_tags[port.name + "_out"] = tag
                with dpg.node_attribute(label=port.name, tag=tag,
                                        attribute_type=dpg.mvNode_Attr_Output):
                    dpg.add_text(port.name)

    # Registra no estado
    state = GlyphState(
        glyph_id=glyph_id,
        func=func,
        library=glyph_def.library,
        pos_x=pos_x,
        pos_y=pos_y,
        params={p.name: p.default for p in glyph_def.params},
        dpg_node_tag=node_tag,
        dpg_attr_tags=attr_tags,
    )
    APP_STATE["glyphs"][glyph_id] = state
    APP_STATE["dirty"] = True
    return glyph_id
```

### 5.3 Callbacks de Conexão

```python
def on_link_created(sender, app_data):
    """Chamado pelo dpg.node_editor quando o usuário conecta dois atributos."""
    output_attr, input_attr = app_data
    link_tag = _new_tag()
    dpg.add_node_link(output_attr, input_attr,
                      parent=NODE_EDITOR_TAG, tag=link_tag)
    APP_STATE["links"][link_tag] = (output_attr, input_attr)
    APP_STATE["dirty"] = True

def on_link_deleted(sender, app_data):
    """Chamado quando o usuário deleta uma conexão (ctrl+click)."""
    link_tag = app_data
    if link_tag in APP_STATE["links"]:
        del APP_STATE["links"][link_tag]
    dpg.delete_item(link_tag)
    APP_STATE["dirty"] = True
```

### 5.4 Estado Visual (cores durante execução)

```python
# Paleta de cores por estado (tema Cantata)
GLYPH_COLORS = {
    "ready":   (80, 80, 80, 255),    # cinza
    "running": (30, 30, 30, 255),    # preto
    "done":    (220, 220, 220, 255), # branco/claro
    "error":   (180, 30, 30, 255),   # vermelho
}

def set_glyph_status(glyph_id: str, status: str):
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return
    state.status = status
    color = GLYPH_COLORS[status]
    # Dear PyGui: alterar tema do nó específico via dpg.set_item_theme
    # ou via mvThemeColor para o node_tag
    theme_tag = _get_or_create_node_theme(color)
    dpg.bind_item_theme(state.dpg_node_tag, theme_tag)
```

### 5.5 Remover Glyph

```python
def remove_glyph(glyph_id: str):
    state = APP_STATE["glyphs"].get(glyph_id)
    if not state:
        return
    # Remove links conectados a este nó
    links_to_remove = [
        lid for lid, (out, inp) in APP_STATE["links"].items()
        if out in state.dpg_attr_tags.values()
        or inp in state.dpg_attr_tags.values()
    ]
    for lid in links_to_remove:
        dpg.delete_item(lid)
        del APP_STATE["links"][lid]
    # Remove o nó do canvas
    dpg.delete_item(state.dpg_node_tag)
    del APP_STATE["glyphs"][glyph_id]
    APP_STATE["dirty"] = True
```

---

## 6. `wksp_io.py` — Serialização .wksp

### 6.1 Salvar (canvas → arquivo)

```python
# gui/wksp_io.py
import os
from gui.app import APP_STATE

def save_wksp(path: str):
    """Serializa o estado atual do canvas para o formato .wksp."""
    lines = []
    lines.append("WorkspaceBegin: 1.0\n")
    lines.append("\nVariablesBegin:\n")
    lines.append("\nVariablesEnd:\n")

    for glyph_id, state in APP_STATE["glyphs"].items():
        pos = dpg.get_item_pos(state.dpg_node_tag)
        x, y = int(pos[0]), int(pos[1])

        params_str = ""
        for k, v in state.params.items():
            if v:
                if " " in str(v) or v.startswith("["):
                    params_str += f" -{k} '{v}'"
                else:
                    params_str += f" -{k} {v}"

        lines.append(
            f"Glyph:{state.library}:{state.func}::localhost:{glyph_id}:{x}:{y}::{params_str}\n"
        )

    lines.append("\n")
    # Mapeia attr_tag → (glyph_id, port_name) para gerar NodeConnections
    tag_to_port = {}
    for gid, state in APP_STATE["glyphs"].items():
        for port_key, tag in state.dpg_attr_tags.items():
            tag_to_port[tag] = (gid, port_key)

    for lid, (out_tag, in_tag) in APP_STATE["links"].items():
        if out_tag in tag_to_port and in_tag in tag_to_port:
            out_gid, out_port = tag_to_port[out_tag]
            in_gid, in_port  = tag_to_port[in_tag]
            # Remove sufixo "_out" / "_in"
            out_port_name = out_port.replace("_out", "")
            in_port_name  = in_port.replace("_in", "")
            lines.append(
                f"NodeConnection:data:{out_gid}:{out_port_name}:{in_gid}:{in_port_name}\n"
            )

    lines.append("\nWorkspaceEnd: 1.0\n")

    with open(path, "w") as f:
        f.writelines(lines)

    APP_STATE["wksp_path"] = path
    APP_STATE["dirty"] = False
```

### 6.2 Carregar (arquivo → canvas)

```python
def load_wksp(path: str):
    """
    Usa readWorkflow.fileRead() para parsear o .wksp,
    depois reconstrói os Glyphs e conexões no canvas via add_glyph_to_canvas().
    """
    import sys
    sys.argv = ["gui", path]  # readWorkflow lê sys.argv[1]
    from readWorkflow import Workspace, fileRead
    from gui.canvas import add_glyph_to_canvas, connect_ports

    # Limpa canvas atual
    clear_canvas()

    workspace = Workspace()
    fileRead(workspace)

    # Reconstrói Glyphs
    tag_by_glyph_and_port = {}
    for glyph in workspace.lstGlyph:
        params = {p.getName(): p.getValue() for p in glyph.lst_par}
        glyph_id = add_glyph_to_canvas(
            func=glyph.func,
            pos_x=float(glyph.glyph_x),
            pos_y=float(glyph.glyph_y),
            forced_id=glyph.glyph_id,
            params=params,
        )

    # Reconstrói conexões
    for conn in workspace.lstConnections:
        out_gid = conn.output_glyph_id
        out_port = conn.output_varname
        for inp in conn.lst_con_input:
            in_gid = inp.Par_glyph_id
            in_port = inp.Par_name
            connect_ports(out_gid, out_port, in_gid, in_port)

    APP_STATE["wksp_path"] = path
    APP_STATE["dirty"] = False
```

---

## 7. `toolbar.py` — Menus e Execução

### 7.1 Menus Dear PyGui

```python
# gui/toolbar.py
import dearpygui.dearpygui as dpg
from gui.wksp_io import save_wksp, load_wksp
from gui.runner import run_workflow, stop_workflow

def setup_menu_bar():
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Novo",        callback=on_new,       shortcut="Ctrl+N")
            dpg.add_menu_item(label="Abrir...",    callback=on_open,      shortcut="Ctrl+O")
            dpg.add_menu_item(label="Salvar",      callback=on_save,      shortcut="Ctrl+S")
            dpg.add_menu_item(label="Salvar Como", callback=on_save_as,   shortcut="Ctrl+Shift+S")
            dpg.add_separator()
            dpg.add_menu_item(label="Sair",        callback=dpg.stop_dearpygui)

        with dpg.menu(label="Run"):
            dpg.add_menu_item(label="Executar (GPU)", callback=lambda: run_workflow("GPU"), shortcut="F5")
            dpg.add_menu_item(label="Executar (CPU)", callback=lambda: run_workflow("CPU"))
            dpg.add_menu_item(label="Parar",          callback=stop_workflow, shortcut="F6")

        with dpg.menu(label="View"):
            dpg.add_menu_item(label="Fit to Screen", callback=on_fit_screen, shortcut="Ctrl+Shift+F")
            dpg.add_menu_item(label="Toggle Log",    callback=on_toggle_log)
```

---

## 8. `runner.py` — Execução via Subprocess

Módulo separado para isolar a lógica de execução:

```python
# gui/runner.py
import subprocess
import threading
import re
import tempfile
import os
from gui.app import APP_STATE
from gui.canvas import set_glyph_status
from gui.log_panel import append_log
from gui.wksp_io import save_wksp

# Regex para detectar início de execução de um Glyph no stdout do executor
# Padrão atual: "A função vglClConvolution está sendo executada"
_FUNC_START_RE = re.compile(r"A função (\S+) está sendo executada")

def run_workflow(device: str = "GPU"):
    if APP_STATE["exec_running"]:
        return

    # 1. Salva estado atual em arquivo temporário
    tmp = tempfile.NamedTemporaryFile(suffix=".wksp", delete=False, mode="w")
    tmp.close()
    save_wksp(tmp.name)

    # 2. Reseta status visual de todos os Glyphs
    for gid in APP_STATE["glyphs"]:
        set_glyph_status(gid, "ready")

    # 3. Sobe processo em thread separada
    APP_STATE["exec_running"] = True
    APP_STATE["exec_process"] = None

    def _run():
        env = os.environ.copy()
        if device == "CPU":
            env["LD_LIBRARY_PATH"] = "/opt/AMDAPPSDK-2.9-1/lib/x86_64/"
        else:
            env["LD_LIBRARY_PATH"] = "/opt/amdgpu/lib/x86_64-linux-gnu/:/opt/rocm/lib/"

        proc = subprocess.Popen(
            ["python3", "execWorkflow.py", tmp.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        APP_STATE["exec_process"] = proc

        # Glyph atual em execução (para marcar RUNNING → DONE)
        current_func = None
        current_gid  = None

        for line in proc.stdout:
            line = line.rstrip()
            append_log(line)  # envia para o painel de log

            m = _FUNC_START_RE.search(line)
            if m:
                # Marca glyph anterior como DONE
                if current_gid:
                    set_glyph_status(current_gid, "done")

                func_name = m.group(1)
                # Encontra o glyph_id correspondente (pelo func name, em ordem)
                current_gid = _find_next_glyph_by_func(func_name, current_gid)
                current_func = func_name
                if current_gid:
                    set_glyph_status(current_gid, "running")

        proc.wait()

        # Marca último glyph como done
        if current_gid:
            set_glyph_status(current_gid, "done" if proc.returncode == 0 else "error")

        APP_STATE["exec_running"] = False
        APP_STATE["exec_process"] = None
        os.unlink(tmp.name)

        status = "OK" if proc.returncode == 0 else f"ERRO (código {proc.returncode})"
        append_log(f"\n[Execução finalizada: {status}]")

    threading.Thread(target=_run, daemon=True).start()


def stop_workflow():
    proc = APP_STATE.get("exec_process")
    if proc:
        proc.terminate()
        APP_STATE["exec_running"] = False
        append_log("[Execução interrompida pelo usuário]")


def _find_next_glyph_by_func(func_name: str, after_gid: str | None) -> str | None:
    """
    Retorna o glyph_id do próximo Glyph com func == func_name
    que ainda não está DONE.
    """
    for gid, state in APP_STATE["glyphs"].items():
        if state.func == func_name and state.status not in ("done", "error"):
            return gid
    return None
```

---

## 9. `log_panel.py` — Painel de Log

```python
# gui/log_panel.py
import dearpygui.dearpygui as dpg
import threading

LOG_TAG = 200
_lock = threading.Lock()

def setup_log_panel():
    with dpg.child_window(tag=LOG_TAG, height=150, border=True):
        dpg.add_text("", tag=201, wrap=0)  # texto do log

def append_log(line: str):
    """Thread-safe: adiciona linha ao log."""
    with _lock:
        # Dear PyGui requer chamadas na thread principal; usar render callback
        # Alternativa: buffer + flush no render loop
        _LOG_BUFFER.append(line)

_LOG_BUFFER = []

def flush_log_buffer():
    """Chamado no render loop principal para atualizar o widget."""
    if _LOG_BUFFER:
        with _lock:
            lines = _LOG_BUFFER.copy()
            _LOG_BUFFER.clear()
        current = dpg.get_value(201) or ""
        dpg.set_value(201, current + "\n".join(lines) + "\n")
```

---

## 10. `app.py` — Ponto de Montagem

```python
# gui/app.py
import dearpygui.dearpygui as dpg
from gui.toolbar import setup_menu_bar
from gui.sidebar_glyphs import setup_sidebar
from gui.canvas import setup_canvas
from gui.log_panel import setup_log_panel, flush_log_buffer

def run():
    dpg.create_context()

    # Tema escuro (padrão do Dear PyGui)
    dpg.create_viewport(title="VGLGui", width=1400, height=900)
    dpg.setup_dearpygui()

    with dpg.window(label="VGLGui", tag=1, width=1400, height=900,
                    no_title_bar=True, no_move=True, no_resize=True):
        setup_menu_bar()

        with dpg.group(horizontal=True):
            # Sidebar esquerda — largura fixa
            with dpg.child_window(width=200, border=True):
                setup_sidebar()

            # Canvas — ocupa o restante
            with dpg.child_window(border=True):
                setup_canvas()

        # Log inferior
        setup_log_panel()

    dpg.set_primary_window(1, True)
    dpg.show_viewport()

    # Render loop
    while dpg.is_dearpygui_running():
        flush_log_buffer()   # flush thread-safe do log
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
```

---

## 11. `main.py` — Entry Point

```python
# gui/main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gui.app import run

if __name__ == "__main__":
    run()
```

---

## 12. Dependências

```
dearpygui>=1.11.0
```

As demais dependências (`pyopencl`, `numpy`, `matplotlib`, `scikit-image`) já existem no ambiente virtual `my_env`.

Instalação:
```bash
source my_env/bin/activate
pip install dearpygui
```

---

## 13. Limitações Conhecidas (v1)

| Item | Detalhe |
|------|---------|
| Feedback de RUNNING por Glyph | Depende do padrão de log `"A função X está sendo executada"` — funções sem esse print não serão detectadas |
| `vl.vglClInit(GPU)` no topo de execWorkflow.py | Inicializa OpenCL no import — mitigado rodando como subprocess |
| `sys.argv` em readWorkflow.py | Contornado ao setar `sys.argv` antes do import em `load_wksp()` |
| Ciclos no grafo | Validação não implementada na v1 — deixado para v2 |
| Undo/Redo | Não implementado na v1 |
