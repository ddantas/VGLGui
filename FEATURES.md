# VGLGui — Funcionalidades Implementadas

Editor visual de workflows de processamento de imagem baseado em VisionGL (OpenCL).

---

## 1. Canvas / Editor de Nós

- Node editor visual (Dear PyGui `dpg.node_editor`)
- Minimapa no canto inferior direito
- **Zoom** com `Ctrl + Scroll` (faixa 0.1× – 5.0×, pivot no centro do canvas)
- **Pan** com botão do meio do mouse (nativo do DPG)
- Adição de nós via sidebar (paleta de funções)
- Exclusão de nó com botão **X** no próprio nó
- Expand/collapse de parâmetros com botão **=**
- Popup de informação do nó com botão **i** (ports, parâmetros, categoria)
- Conexão de portas por arraste (drag & drop)
- Desconexão de links por clique direito
- Feedback visual de status do nó durante execução:
  | Status   | Cor             |
  |----------|-----------------|
  | Ready    | Cinza           |
  | Running  | Amarelo-laranja  |
  | Done     | Verde escuro    |
  | Error    | Vermelho        |

---

## 2. Auto Layout

Dois algoritmos acessíveis via menu **View**:

### Left → Right (padrão, `Ctrl+L`)
- Cálculo de nível por dependência (algoritmo de Kahn + longest-path)
- Nós `vglCreateImage` posicionados **acima** do seu consumidor (mesma coluna)
- Nós terminais (`vglSaveImage`, `ShowImage`) posicionados **abaixo** do predecessor
- Espaçamento compacto (280 px horizontal, 200 px vertical)

### Top → Bottom
- Grade multi-coluna (3 colunas por padrão)
- Organização por nível de dependência

---

## 3. Paleta de Funções (Sidebar)

- Campo de **busca** em tempo real (case-insensitive)
- **11 categorias** colapsáveis:
  - I/O, Allocation, Color, Filters, Morphology, Operations, Special, ND, 3D, Fuzzy 2D, Fuzzy 3D
- **70+ funções VGL** registradas, incluindo:
  - I/O: `vglLoadImage`, `vglLoad2dImage`, `vglLoad3dImage`, `vglLoadNdImage`, `vglSaveImage`, `ShowImage`
  - Morfologia: `vglClDilate`, `vglClErode`, `Closing`, `blackhat`, variantes N e 3D
  - Filtros: `vglClConvolution`, `vglClBlurSq3`, `vglClNConvolution`
  - Operações: `vglClSub`, `vglClSum`, `vglClMax`, `vglClMin`, `vglClThreshold`
  - Fuzzy 2D e 3D: 16 variantes cada (Alg, Arith, Bound, DaP, Drastic, Geo, Hamacher, Std)
  - ND: `vglClNdConvolution`, `vglClNdDilate`, `vglClNdErode`, `vglClNdThreshold`, etc.
  - 3D: `vglCl3dDilate`, `vglCl3dErode`, `vglCl3dConvolution`, etc.
  - Especiais: `Reconstruct`, `vglShape`, `vglStrel`
- Botão **Nova Procedure** no topo da sidebar

---

## 4. Widgets de Parâmetros

| Tipo       | Widget                                      |
|------------|---------------------------------------------|
| `text`     | Input de texto                              |
| `int`      | Input numérico inteiro (com min/max)        |
| `float`    | Input numérico decimal                      |
| `bool`     | Checkbox                                    |
| `file`     | Input de texto + botão `...` (file dialog) |
| `array`    | Input de texto (notação `[1,0,1,...]`)     |
| `strel`    | Gerador de elemento estruturante 2D        |
| `strel3d`  | Gerador de elemento estruturante 3D        |

**Gerador de elemento estruturante (strel):**
- Formas: Rectangle, Cross, Diamond, Disc (2D) / Box (3D)
- Dimensões W, H (e Z para 3D) configuráveis
- Feedback com contagem de pixels ativos após gerar

---

## 5. Procedures (Sub-Workflows)

- Procedure aparece no canvas principal como nó **azul** `[P] Nome`
- Portas `i` (entrada) e `o` (saída) no nó principal
- Botão **Abrir** abre popup com editor de nós interno (900×650 px)
- Popup contém:
  - **ExternalInput** (verde) — porta de entrada da procedure
  - **ExternalOutput** (vermelho) — porta de saída da procedure
  - Todos os widgets e funções do canvas principal
- Quando popup está aberto, a sidebar adiciona nós à procedure ativa
- Procedures salvas e carregadas no formato `.wksp`
- Suporte a formato legado (ExtInput no workspace principal)

---

## 6. Menus e Atalhos de Teclado

### File
| Ação       | Atalho          |
|------------|-----------------|
| New        | `Ctrl+N`        |
| Open       | `Ctrl+O`        |
| Save       | `Ctrl+S`        |
| Save As    | `Ctrl+Shift+S`  |
| Exit       | —               |

### Run
| Ação       | Atalho |
|------------|--------|
| Run GPU    | `F5`   |
| Run CPU    | —      |
| Stop       | `F6`   |

### View
| Ação              | Atalho   |
|-------------------|----------|
| Layout L→R        | `Ctrl+L` |
| Layout T→B        | —        |
| Toggle Log Panel  | —        |

### Exemplos (12 workflows pré-construídos)
| Categoria   | Workflows                                              |
|-------------|--------------------------------------------------------|
| 2D          | Conv+Dilate+Erode, Gray, RGB, Drive Segmentation       |
| Fundus      | Fundus Retina, Fundus v2                               |
| 3D          | 3D Demo, 3D Functions                                  |
| ND          | ND Basic, ND Total, ND Strel (type), ND Strel (window) |
| Procedures  | GrayscaleConvert, Demo Fundus                          |

### Language
- Português (PT-BR) — padrão
- English

### Outros atalhos
| Ação               | Atalho         |
|--------------------|----------------|
| Deletar nó(s)      | `Delete`       |
| Zoom in/out canvas | `Ctrl+Scroll`  |

---

## 7. Execução de Workflows

- Execução em **thread de fundo** (não bloqueia a UI)
- Seleção de dispositivo: **GPU** ou **CPU**
- Configuração automática de `LD_LIBRARY_PATH` por dispositivo
- Arquivo `.wksp` temporário gerado antes da execução
- Monitoramento em tempo real via stdout do subprocesso
- **Status visual** dos nós atualizado conforme execução progride
- Regex para detectar início de cada função: `A função <FUNC> está sendo executada`
- Botão **Stop** encerra o processo imediatamente

---

## 8. Preview de Imagens

- Função `ShowImage` emite `[GUI_SHOW] <caminho>` no stdout
- Imagem carregada com PIL (conversão RGBA → float32)
- Redimensionamento automático (máx. 700 px na dimensão maior)
- Popup dedicado com título = nome do arquivo
- Limpeza de textura ao fechar popup

---

## 9. Painel de Log

- Exibe saída em tempo real da execução
- Botão **Limpar** — limpa o log
- Botão **Copiar** — copia todo o conteúdo para a área de transferência
- Auto-scroll para o fim a cada nova mensagem
- Thread-safe (buffer com lock)

---

## 10. Salvar / Carregar Workspace (`.wksp`)

### Salvar
- Serialização completa: glyphs, parâmetros, posições, conexões, procedures
- Formato compatível com o interpretador `execWorkflow.py`

### Carregar
- Reconstrução completa do estado visual
- Suporte a procedures (formato atual e legado)
- Detecção de nós sobrepostos → dispara auto layout automaticamente
- Relatório no log: quantidade de glyphs, links e procedures carregados

---

## 11. Internacionalização (i18n)

- **PT-BR** (padrão) e **English**
- 90+ chaves de tradução cobrindo menus, botões, mensagens e categorias
- Troca de idioma via menu Language → salva em `config.json` → reinicia a UI
- Interpolação de variáveis: `t("chave", var=valor)`

---

## 12. Configuração Persistente

- Arquivo `gui/config.json`
- Salva: idioma selecionado
- Carregado automaticamente na inicialização

---

## Resumo Quantitativo

| Item                          | Quantidade |
|-------------------------------|------------|
| Funções VGL registradas       | 70+        |
| Exemplos pré-construídos      | 12         |
| Categorias na sidebar         | 11         |
| Tipos de widgets de parâmetro | 8          |
| Estados visuais de nó         | 4          |
| Chaves de tradução            | 90+        |
| Idiomas suportados            | 2          |
| Atalhos de teclado            | 7+         |
