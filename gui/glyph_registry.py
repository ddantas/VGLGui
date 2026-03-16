from dataclasses import dataclass, field


@dataclass
class PortDef:
    name: str
    kind: str       # "input" | "output"
    required: bool = True


@dataclass
class ParamDef:
    name: str
    type: str       # "text" | "int" | "float" | "array" | "file" | "bool"
    default: str = ""
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name


@dataclass
class GlyphDef:
    func: str
    label: str
    category: str
    ports: list
    params: list
    library: str = "VGL_CL"


# ---------------------------------------------------------------------------
# Helpers para reduzir repetição
# ---------------------------------------------------------------------------

def _img_in_out(extra_params=None):
    """Padrão mais comum: img_input(in) + img_output(in) + img_output(out)."""
    return (
        [PortDef("img_input", "input"), PortDef("img_output", "input"), PortDef("img_output", "output")],
        extra_params or [],
    )

def _binary_op():
    """Duas entradas de imagem: img_input1, img_input2, img_output."""
    return (
        [
            PortDef("img_input1", "input"),
            PortDef("img_input2", "input"),
            PortDef("img_output", "input"),
            PortDef("img_output", "output"),
        ],
        [],
    )

def _morph_params_2d():
    return [
        ParamDef("convolution_window", "strel",  default="[1,1,1,1,1,1,1,1,1]"),
        ParamDef("window_size_x",      "hidden", default="3"),
        ParamDef("window_size_y",      "hidden", default="3"),
    ]

def _n_morph_params_2d():
    return [
        ParamDef("convolution_window", "strel",  default="[1,1,1,1,1,1,1,1,1]"),
        ParamDef("window_size_x",      "hidden", default="3"),
        ParamDef("window_size_y",      "hidden", default="3"),
        ParamDef("n", "int", default="1", label="n (repetitions)"),
    ]

def _morph_params_3d():
    return [
        ParamDef("convolution_window", "strel3d", default="[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"),
        ParamDef("window_size_x",      "hidden",  default="3"),
        ParamDef("window_size_y",      "hidden",  default="3"),
        ParamDef("window_size_z",      "hidden",  default="3"),
    ]

def _fuzzy_2d(func, category="Fuzzy 2D"):
    ports, _ = _img_in_out()
    return GlyphDef(func=func, label=func, category=category, ports=ports,
                    params=_morph_params_2d())

def _fuzzy_3d(func):
    ports, _ = _img_in_out()
    return GlyphDef(func=func, label=func, category="Fuzzy 3D", ports=ports,
                    params=_morph_params_3d())




# Category base colors (R, G, B) — used for node themes
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "I/O":         (50,  100, 160),
    "Allocation":  (70,  70,  90),
    "Color":       (120, 55,  125),
    "Filters":     (45,  105, 110),
    "Morphology":  (60,  105, 50),
    "Operations":  (125, 85,  35),
    "Special":     (85,  55,  135),
    "ND":          (45,  65,  135),
    "3D":          (38,  105, 128),
    "Fuzzy 2D":    (125, 65,  55),
    "Fuzzy 3D":    (105, 50,  50),
    "Procedures":  (30,  85,  90),
    "CV":          (30,  110, 100),
}


# ---------------------------------------------------------------------------
# Registro completo
# ---------------------------------------------------------------------------

GLYPH_REGISTRY: dict[str, GlyphDef] = {

    # ── I/O ─────────────────────────────────────────────────────────────────

    "vglLoadImage": GlyphDef(
        func="vglLoadImage", label="vglLoadImage", category="I/O",
        ports=[PortDef("RETVAL", "output")],
        params=[
            ParamDef("filename",  "file",  label="filename"),
            ParamDef("iscolor",   "bool",  default="1"),
            ParamDef("has_mipmap","bool",  default="0"),
        ],
    ),
    "vglLoad2dImage": GlyphDef(
        func="vglLoad2dImage", label="vglLoadImage (2D)", category="I/O",
        ports=[PortDef("RETVAL", "output")],
        params=[
            ParamDef("filename",  "file",  label="filename"),
            ParamDef("iscolor",   "bool",  default="1"),
            ParamDef("has_mipmap","bool",  default="0"),
        ],
    ),
    "vglLoad3dImage": GlyphDef(
        func="vglLoad3dImage", label="vglLoadImage (3D)", category="I/O",
        ports=[PortDef("RETVAL", "output")],
        params=[
            ParamDef("filename",  "file",  label="filename"),
            ParamDef("iscolor",   "bool",  default="1"),
            ParamDef("has_mipmap","bool",  default="0"),
        ],
    ),
    "vglLoadNdImage": GlyphDef(
        func="vglLoadNdImage", label="vglLoadImage (ND)", category="I/O",
        ports=[PortDef("RETVAL", "output")],
        params=[
            ParamDef("filename",  "file",  label="filename"),
            ParamDef("iscolor",   "bool",  default="1"),
            ParamDef("has_mipmap","bool",  default="0"),
        ],
    ),
    "vglSaveImage": GlyphDef(
        func="vglSaveImage", label="vglSaveImage", category="I/O",
        ports=[PortDef("image", "input")],
        params=[ParamDef("filename", "file", label="filename")],
    ),
    "ShowImage": GlyphDef(
        func="ShowImage", label="ShowImage", category="I/O",
        ports=[PortDef("image", "input")],
        params=[],
    ),

    # ── Alocação ─────────────────────────────────────────────────────────────

    "vglCreateImage": GlyphDef(
        func="vglCreateImage", label="vglCreateImage", category="Allocation",
        ports=[PortDef("img", "input"), PortDef("RETVAL", "output")],
        params=[],
    ),

    # ── Cor ──────────────────────────────────────────────────────────────────

    "vglClRgb2Gray": GlyphDef(
        func="vglClRgb2Gray", label="vglClRgb2Gray", category="Color",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglClSwapRgb": GlyphDef(
        func="vglClSwapRgb", label="vglClSwapRgb", category="Color",
        ports=[PortDef("src","input"), PortDef("dst","input"),
               PortDef("dst","output")],
        params=[],
    ),

    # ── Filtros ───────────────────────────────────────────────────────────────

    "vglClConvolution": GlyphDef(
        func="vglClConvolution", label="vglClConvolution", category="Filters",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[
            ParamDef("convolution_window","array",
                     default="[0,0,0.125,0,0.125,0.5,0.125,0,0.125,0,0]"),
            ParamDef("window_size_x","int", default="3"),
            ParamDef("window_size_y","int", default="3"),
        ],
    ),
    "vglClBlurSq3": GlyphDef(
        func="vglClBlurSq3", label="vglClBlurSq3", category="Filters",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglClCopy": GlyphDef(
        func="vglClCopy", label="vglClCopy", category="Filters",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglClInvert": GlyphDef(
        func="vglClInvert", label="vglClInvert", category="Filters",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),

    # ── Morfologia ────────────────────────────────────────────────────────────

    "vglClDilate": GlyphDef(
        func="vglClDilate", label="vglClDilate", category="Morphology",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(),
    ),
    "vglClErode": GlyphDef(
        func="vglClErode", label="vglClErode", category="Morphology",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(),
    ),
    "Closing": GlyphDef(
        func="Closing", label="Closing", category="Morphology",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(),
    ),
    "blackhat": GlyphDef(
        func="blackhat", label="blackhat", category="Morphology",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(),
    ),
    "vglClNDilate": GlyphDef(
        func="vglClNDilate", label="vglClNDilate", category="Morphology",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_n_morph_params_2d(),
    ),
    "vglClNErode": GlyphDef(
        func="vglClNErode", label="vglClNErode", category="Morphology",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_n_morph_params_2d(),
    ),
    "vglClNConvolution": GlyphDef(
        func="vglClNConvolution", label="vglClNConvolution", category="Filters",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_n_morph_params_2d(),
    ),

    # ── Operações ─────────────────────────────────────────────────────────────

    "vglClSub": GlyphDef(
        func="vglClSub", label="vglClSub", category="Operations",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClSum": GlyphDef(
        func="vglClSum", label="vglClSum", category="Operations",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClMax": GlyphDef(
        func="vglClMax", label="vglClMax", category="Operations",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClMin": GlyphDef(
        func="vglClMin", label="vglClMin", category="Operations",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClThreshold": GlyphDef(
        func="vglClThreshold", label="vglClThreshold", category="Operations",
        ports=[PortDef("src","input"), PortDef("dst","input"),
               PortDef("dst","output")],
        params=[ParamDef("thresh","float", default="0.5")],
    ),

    # ── Especiais ─────────────────────────────────────────────────────────────

    "Reconstruct": GlyphDef(
        func="Reconstruct", label="Reconstruct", category="Special",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(),
    ),
    "vglShape": GlyphDef(
        func="vglShape", label="vglShape", category="Special",
        ports=[PortDef("img_output","input")],
        params=[
            ParamDef("width",  "int", default="3"),
            ParamDef("height", "int", default="3"),
        ],
    ),
    "vglStrel": GlyphDef(
        func="vglStrel", label="vglStrel", category="Special",
        ports=[PortDef("shape","input")],
        params=[
            ParamDef("type_or_window","text", default="gaussian",
                     label="type / window"),
            ParamDef("ndim","int", default="2"),
        ],
    ),
    "External Input (1)": GlyphDef(
        func="External Input (1)", label="External Input", category="Special",
        ports=[PortDef("i","input"), PortDef("o","output")],
        params=[],
    ),
    "External Output (1)": GlyphDef(
        func="External Output (1)", label="External Output", category="Special",
        ports=[PortDef("o","input")],
        params=[],
    ),

    # ── Procedures ────────────────────────────────────────────────────────────
    "ProcedureBegin": GlyphDef(
        func="ProcedureBegin", label="Procedure",
        category="Procedures",
        ports=[PortDef("i", "input"), PortDef("o", "output")],
        params=[],
        library="VGL_GUI",
    ),

    # ── ND ────────────────────────────────────────────────────────────────────

    "vglClNdConvolution": GlyphDef(
        func="vglClNdConvolution", label="vglClNdConvolution", category="ND",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("window","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClNdCopy": GlyphDef(
        func="vglClNdCopy", label="vglClNdCopy", category="ND",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglClNdDilate": GlyphDef(
        func="vglClNdDilate", label="vglClNdDilate", category="ND",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("window","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClNdErode": GlyphDef(
        func="vglClNdErode", label="vglClNdErode", category="ND",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("window","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglClNdNot": GlyphDef(
        func="vglClNdNot", label="vglClNdNot", category="ND",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglClNdThreshold": GlyphDef(
        func="vglClNdThreshold", label="vglClNdThreshold", category="ND",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),

    # ── 3D ────────────────────────────────────────────────────────────────────

    "vglCl3dBlurSq3": GlyphDef(
        func="vglCl3dBlurSq3", label="vglCl3dBlurSq3", category="3D",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dConvolution": GlyphDef(
        func="vglCl3dConvolution", label="vglCl3dConvolution", category="3D",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_3d(),
    ),
    "vglCl3dCopy": GlyphDef(
        func="vglCl3dCopy", label="vglCl3dCopy", category="3D",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dDilate": GlyphDef(
        func="vglCl3dDilate", label="vglCl3dDilate", category="3D",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_3d(),
    ),
    "vglCl3dErode": GlyphDef(
        func="vglCl3dErode", label="vglCl3dErode", category="3D",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_3d(),
    ),
    "vglCl3dMax": GlyphDef(
        func="vglCl3dMax", label="vglCl3dMax", category="3D",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dMin": GlyphDef(
        func="vglCl3dMin", label="vglCl3dMin", category="3D",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dNot": GlyphDef(
        func="vglCl3dNot", label="vglCl3dNot", category="3D",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dSub": GlyphDef(
        func="vglCl3dSub", label="vglCl3dSub", category="3D",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dSum": GlyphDef(
        func="vglCl3dSum", label="vglCl3dSum", category="3D",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[],
    ),
    "vglCl3dThreshold": GlyphDef(
        func="vglCl3dThreshold", label="vglCl3dThreshold", category="3D",
        ports=[PortDef("src","input"), PortDef("dst","input"),
               PortDef("dst","output")],
        params=[ParamDef("thresh","float", default="0.5")],
    ),

    # ── CV (OpenCV) ───────────────────────────────────────────────────────────

    "vglCvBlurSq3": GlyphDef(
        func="vglCvBlurSq3", label="vglCvBlurSq3", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvConvolution": GlyphDef(
        func="vglCvConvolution", label="vglCvConvolution", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(), library="VGL_CV",
    ),
    "vglCvCopy": GlyphDef(
        func="vglCvCopy", label="vglCvCopy", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvDilate": GlyphDef(
        func="vglCvDilate", label="vglCvDilate", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(), library="VGL_CV",
    ),
    "vglCvErode": GlyphDef(
        func="vglCvErode", label="vglCvErode", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=_morph_params_2d(), library="VGL_CV",
    ),
    "vglCvInvert": GlyphDef(
        func="vglCvInvert", label="vglCvInvert", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvMax": GlyphDef(
        func="vglCvMax", label="vglCvMax", category="CV",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvMin": GlyphDef(
        func="vglCvMin", label="vglCvMin", category="CV",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvRgb2Gray": GlyphDef(
        func="vglCvRgb2Gray", label="vglCvRgb2Gray", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvSub": GlyphDef(
        func="vglCvSub", label="vglCvSub", category="CV",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvSum": GlyphDef(
        func="vglCvSum", label="vglCvSum", category="CV",
        ports=[PortDef("img_input1","input"), PortDef("img_input2","input"),
               PortDef("img_output","input"), PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvSwapRgb": GlyphDef(
        func="vglCvSwapRgb", label="vglCvSwapRgb", category="CV",
        ports=[PortDef("img_input","input"), PortDef("img_output","input"),
               PortDef("img_output","output")],
        params=[], library="VGL_CV",
    ),
    "vglCvThreshold": GlyphDef(
        func="vglCvThreshold", label="vglCvThreshold", category="CV",
        ports=[PortDef("src","input"), PortDef("dst","input"),
               PortDef("dst","output")],
        params=[
            ParamDef("thresh", "float", default="0.5"),
            ParamDef("top",    "float", default="1.0"),
        ], library="VGL_CV",
    ),

    # ── Fuzzy 2D ─────────────────────────────────────────────────────────────

    "vglClFuzzyAlgDilate":     _fuzzy_2d("vglClFuzzyAlgDilate"),
    "vglClFuzzyAlgErode":      _fuzzy_2d("vglClFuzzyAlgErode"),
    "vglClFuzzyArithDilate":   _fuzzy_2d("vglClFuzzyArithDilate"),
    "vglClFuzzyArithErode":    _fuzzy_2d("vglClFuzzyArithErode"),
    "vglClFuzzyBoundDilate":   _fuzzy_2d("vglClFuzzyBoundDilate"),
    "vglClFuzzyBoundErode":    _fuzzy_2d("vglClFuzzyBoundErode"),
    "vglClFuzzyDaPDilate":     _fuzzy_2d("vglClFuzzyDaPDilate"),
    "vglClFuzzyDaPErode":      _fuzzy_2d("vglClFuzzyDaPErode"),
    "vglClFuzzyDrasticDilate": _fuzzy_2d("vglClFuzzyDrasticDilate"),
    "vglClFuzzyDrasticErode":  _fuzzy_2d("vglClFuzzyDrasticErode"),
    "vglClFuzzyGeoDilate":     _fuzzy_2d("vglClFuzzyGeoDilate"),
    "vglClFuzzyGeoErode":      _fuzzy_2d("vglClFuzzyGeoErode"),
    "vglClFuzzyHamacherDilate":_fuzzy_2d("vglClFuzzyHamacherDilate"),
    "vglClFuzzyHamacherErode": _fuzzy_2d("vglClFuzzyHamacherErode"),
    "vglClFuzzyStdDilate":     _fuzzy_2d("vglClFuzzyStdDilate"),
    "vglClFuzzyStdErode":      _fuzzy_2d("vglClFuzzyStdErode"),

    # ── Fuzzy 3D ─────────────────────────────────────────────────────────────

    "vglCl3dFuzzyAlgDilate":     _fuzzy_3d("vglCl3dFuzzyAlgDilate"),
    "vglCl3dFuzzyAlgErode":      _fuzzy_3d("vglCl3dFuzzyAlgErode"),
    "vglCl3dFuzzyArithDilate":   _fuzzy_3d("vglCl3dFuzzyArithDilate"),
    "vglCl3dFuzzyArithErode":    _fuzzy_3d("vglCl3dFuzzyArithErode"),
    "vglCl3dFuzzyBoundDilate":   _fuzzy_3d("vglCl3dFuzzyBoundDilate"),
    "vglCl3dFuzzyBoundErode":    _fuzzy_3d("vglCl3dFuzzyBoundErode"),
    "vglCl3dFuzzyDaPDilate":     _fuzzy_3d("vglCl3dFuzzyDaPDilate"),
    "vglCl3dFuzzyDaPErode":      _fuzzy_3d("vglCl3dFuzzyDaPErode"),
    "vglCl3dFuzzyDrasticDilate": _fuzzy_3d("vglCl3dFuzzyDrasticDilate"),
    "vglCl3dFuzzyDrasticErode":  _fuzzy_3d("vglCl3dFuzzyDrasticErode"),
    "vglCl3dFuzzyGeoDilate":     _fuzzy_3d("vglCl3dFuzzyGeoDilate"),
    "vglCl3dFuzzyGeoErode":      _fuzzy_3d("vglCl3dFuzzyGeoErode"),
    "vglCl3dFuzzyHamacherDilate":_fuzzy_3d("vglCl3dFuzzyHamacherDilate"),
    "vglCl3dFuzzyHamacherErode": _fuzzy_3d("vglCl3dFuzzyHamacherErode"),
    "vglCl3dFuzzyStdDilate":     _fuzzy_3d("vglCl3dFuzzyStdDilate"),
    "vglCl3dFuzzyStdErode":      _fuzzy_3d("vglCl3dFuzzyStdErode"),
}

# Ordem de exibição das categorias na sidebar
CATEGORIES: list[str] = [
    "Procedures",
    "I/O",
    "Allocation",
    "Color",
    "Filters",
    "Morphology",
    "Operations",
    "Special",
    "ND",
    "3D",
    "Fuzzy 2D",
    "Fuzzy 3D",
    "CV",
]
