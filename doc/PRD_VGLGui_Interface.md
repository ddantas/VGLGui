# PRD — Interface Visual do VGLGui
**Versão:** 1.1
**Data:** 2026-02-25
**Referência visual:** Cantata — Visual Programming Language for the KHOROS System
**Status:** Rascunho

---

## 1. Contexto e Objetivo

O VGLGui é um interpretador de workflows de processamento de imagens baseado na biblioteca VisionGL (OpenCL). Atualmente o sistema funciona via arquivos `.wksp` que descrevem grafos de funções (Glyphs) e são executados pelo interpretador Python.

O objetivo é criar uma interface gráfica visual no estilo do **Cantata/KHOROS** — um editor de dataflow onde o usuário constrói programas visuais como grafos dirigidos, onde cada nó é uma função e cada arco é um fluxo de dados (imagem).

A interface deve permitir:
- **Construir workflows** posicionando e conectando Glyphs no canvas
- **Configurar parâmetros** de cada Glyph diretamente na interface
- **Executar o workflow** na CPU ou GPU com feedback visual em tempo real
- **Salvar e carregar** arquivos `.wksp` no formato existente

---

## 2. Público-Alvo

Pesquisadores e desenvolvedores de processamento de imagens que precisam construir e testar pipelines de forma iterativa, sem editar arquivos `.wksp` manualmente.

---

## 3. Layout Geral da Aplicação

```
┌─────────────────────────────────────────────────────────────────┐
│  [Toolbar: File | Run | View | ...]                             │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                  │
│   PAINEL     │                                                  │
│   DE         │           CANVAS (área de trabalho)             │
│   GLYPHS     │           grade, pan, zoom                      │
│   (esquerda) │           nós + conexões                        │
│              │                                                  │
│  [categorias]│                                                  │
│  [lista]     │                                                  │
│              │                                                  │
├──────────────┴──────────────────────────────────────────────────┤
│  LOG / SAÍDA (inferior, colapsável)                             │
└─────────────────────────────────────────────────────────────────┘
```

> Nota: No Cantata, os parâmetros são editados diretamente no Glyph expandido no canvas (botão "Expand"), não em uma sidebar separada. O VGLGui seguirá esse mesmo modelo.

---

## 4. Canvas (Área de Trabalho Principal)

### 4.1 Visual
- Fundo com grade de pontos ou linhas finas (cinza escuro)
- Área de trabalho "infinita" — rolagem horizontal e vertical
- Suporte a pan (arrastar fundo) e zoom (scroll)

### 4.2 Comportamentos
| Ação | Interação |
|------|-----------|
| Pan | Clique + arrastar no fundo |
| Zoom | Scroll do mouse |
| Mover Glyph | Arrastar o nó |
| Selecionar Glyph | Clique simples no nó |
| Seleção múltipla | Clique + arrastar no fundo (rubber band) |
| Deletar selecionado | Tecla `Delete` |

---

## 5. Componente — Glyph (Nó)

Inspirado diretamente no design do Cantata: cada Glyph é uma caixa retangular com título, portas de dados (setas/pontos) e botões de controle.

### 5.1 Anatomia do Glyph

```
  ┌─[🗑]──[📋]──[▶]─────────────────────┐
  │         vglClConvolution             │  ← barra de título
  ├──────────────────────────────────────┤
◄─┤ img_input                           │  ← porta de entrada (seta para dentro)
◄─┤ img_output    [imagem alocada]      │
  │ convolution_window  [0,0,0.125,...] │  ← parâmetro (visível no modo compacto)
  │ window_size_x       [ 5 ]          │
  │ window_size_y       [ 5 ]          │
  │                          RETVAL ├─► │  ← porta de saída (seta para fora)
  └──────────────────────────────────────┘
                   [id]                    ← número identificador abaixo
```

### 5.2 Botões de Controle do Glyph (inspirados no Cantata)

Cada Glyph tem **três botões icônicos** na barra de título:

| Botão | Ícone | Ação |
|-------|-------|------|
| **Delete** | 💣 Bomba (como no Cantata original) | Remove o Glyph e todas suas conexões do canvas |
| **Expand/Edit** | 📋 Formulário | Abre/fecha o formulário completo de parâmetros **inline no canvas** |
| **Info** | ℹ️ | Exibe documentação da função (nome, descrição, tipos de entrada/saída) |

> No Cantata original, o botão "Expand" abria a GUI completa do módulo inline. O VGLGui segue esse mesmo padrão: **parâmetros são editados expandindo o Glyph no canvas**, não em painéis externos.

### 5.3 Modos do Glyph

| Modo | Visual | Quando |
|------|--------|--------|
| **Compacto** | Só título + portas | Estado padrão |
| **Expandido** | Título + portas + todos os campos de parâmetro | Após clicar em "Expand" |

### 5.4 Tipos de Porta (Setas)

Seguindo o modelo do Cantata:

| Tipo | Posição | Visual | Comportamento |
|------|---------|--------|---------------|
| **Entrada obrigatória** | Esquerda | Seta preta sólida `◄` | Deve ser conectada para executar |
| **Entrada opcional** | Esquerda | Seta cinza `◄` | Pode ficar desconectada |
| **Saída** | Direita | Seta sólida `►` | Pode conectar a múltiplas entradas |
| **Conectada** | Qualquer | Ponto/dot na ponta | Indica conexão ativa |

### 5.5 Estado Visual do Glyph (feedback de execução — Cantata style)

| Estado | Visual | Significado |
|--------|--------|-------------|
| **READY** (aguardando) | Glyph cinza | Pronto, aguardando execução |
| **RUNNING** (executando) | Glyph preto/escuro | Em execução no momento |
| **DONE** (concluído) | Glyph branco/claro | Executado com sucesso |
| **ERROR** | Glyph vermelho / borda vermelha | Falhou na execução |

### 5.6 Campos de Parâmetro (modo expandido)

| Tipo de Parâmetro | Widget |
|-------------------|--------|
| Caminho de arquivo (`filename`) | Input de texto + botão "Browse" |
| Array (`convolution_window`) | Input de texto multiline |
| Inteiro (`window_size_x/y`) | Input numérico com spinner |
| Booleano (`iscolor`, `has_mipmap`) | Checkbox |
| String simples | Input de texto |

---

## 6. Componente — Conexão (Arco de Dados)

Inspirado no Cantata: linhas vermelhas conectando saídas a entradas.

### 6.1 Visual
- **Cor:** vermelho (`#cc0000`) — linha de fluxo de dados, como no Cantata
- **Estilo:** linha reta ou curva bezier entre os pontos de porta
- **Espessura:** 2px
- **Ponto nas extremidades:** pequeno dot marcando onde a linha toca a porta (como no Cantata)
- **Hover:** destaca em vermelho mais vivo
- **Selecionada:** linha branca; `Delete` remove

### 6.2 Criação de Conexão
1. Clicar em uma porta de saída `►` → linha "fantasma" segue o cursor
2. Clicar em uma porta de entrada `◄` compatível → conexão criada
3. Soltar em local inválido → cancelado

### 6.3 Validação
- Porta de entrada obrigatória aceita **uma** conexão (substituir pede confirmação)
- Não são permitidos ciclos
- Tipos incompatíveis são rejeitados visualmente (porta fica vermelha no hover)

---

## 7. Painel de Glyphs (Sidebar Esquerda)

Inspirado nos **10 botões verdes com pop-up menus** do Cantata, adaptado para Dear PyGui.

### 7.1 Estrutura
Painel fixo à esquerda com botões por categoria. Cada botão abre uma lista de funções disponíveis.

### 7.2 Categorias e Funções

| Botão/Categoria | Funções disponíveis |
|-----------------|---------------------|
| **I/O** | `vglLoadImage`, `vglSaveImage`, `ShowImage` |
| **Alocação** | `vglCreateImage` |
| **Cor** | `vglClRgb2Gray`, `vglClSwapRgb` |
| **Morfologia** | `vglClDilate`, `vglClErode`, `Closing` |
| **Filtros** | `vglClConvolution` |
| **Operações** | `vglCISub`, `vglCIThreshold` |
| **Reconstrução** | `Reconstruct` |

### 7.3 Comportamento
- Clicar em uma função do painel → Glyph é adicionado ao centro do canvas visível
- Arrastar do painel → Glyph posicionado onde for solto no canvas
- Campo de busca no topo para filtrar por nome

---

## 8. Toolbar Superior

### 8.1 Menus

| Menu | Itens |
|------|-------|
| **File** | Novo, Abrir `.wksp`, Salvar, Salvar Como, Sair |
| **Run** | Executar (CPU), Executar (GPU), Parar, Selecionar dispositivo |
| **View** | Fit to Screen, Zoom In, Zoom Out, Toggle Log Panel |
| **Help** | Documentação, Sobre |

### 8.2 Atalhos de Teclado

| Ação | Atalho |
|------|--------|
| Novo | `Ctrl+N` |
| Abrir | `Ctrl+O` |
| Salvar | `Ctrl+S` |
| Salvar Como | `Ctrl+Shift+S` |
| Executar | `F5` |
| Parar | `F6` |
| Desfazer | `Ctrl+Z` |
| Refazer | `Ctrl+Y` |
| Fit to Screen | `Ctrl+Shift+F` |
| Deletar selecionado | `Delete` |

---

## 9. Painel de Log / Saída (Inferior)

### 9.1 Descrição
Painel colapsável na parte inferior. Exibe a saída do interpretador Python em tempo real.

### 9.2 Conteúdo
- Log de texto com timestamp por linha
- Cores: INFO (branco), WARNING (amarelo), ERROR (vermelho)
- Botões: "Limpar", "Copiar"
- Ao final da execução: exibe tempo total e status (OK / ERRO)

---

## 10. Modelo de Dados — Formato `.wksp`

O canvas serializa/deserializa no formato `.wksp` existente:

```
WorkspaceBegin: 1.0

VariablesBegin:
  width_size = 2342
  height_size = 1144
VariablesEnd:

Glyph:VGL_CL:vglLoad2dImage::localhost:1:302:82:: -filename 'path/img.jpg' -iscolor 1
Glyph:VGL_CL:vglCreateImage::localhost:2:562:122::
Glyph:VGL_CL:vglClRgb2Gray::localhost:3:382:182::

NodeConnection:data:1:RETVAL:2:img
NodeConnection:data:1:RETVAL:3:img_input
NodeConnection:data:2:RETVAL:3:img_output

WorkspaceEnd: 1.0
```

**Mapeamento Canvas → Arquivo:**
- Posição X/Y do nó no canvas → campos de coordenada no Glyph
- Parâmetros editados → argumentos `-param value` no Glyph
- Conexões visuais → linhas `NodeConnection`

---

## 11. Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Framework GUI | **Dear PyGui** (`dearpygui`) |
| Editor de nós | `dpg.node_editor` (nativo do Dear PyGui) |
| Lógica de workflow | Reutiliza `readWorkflow.py` e `execWorkflowGen.py` existentes |
| Execução de imagens | `pyopencl`, `vgl_lib` (já presentes no projeto) |
| Linguagem | Python 3.10+ |

### Estrutura de Arquivos Proposta
```
VGLGui/
├── gui/
│   ├── main.py              # Ponto de entrada da aplicação
│   ├── app.py               # Setup Dear PyGui (viewport, tema, layout)
│   ├── canvas.py            # Node editor: glyphs, conexões, interação
│   ├── sidebar_glyphs.py    # Painel esquerdo: paleta de funções VGL
│   ├── toolbar.py           # Menu superior e atalhos
│   ├── log_panel.py         # Painel inferior de logs
│   ├── glyph_registry.py    # Catálogo de funções VGL (portas, params, tipos)
│   └── wksp_io.py           # Leitura/escrita de .wksp
├── readWorkflow.py          # (existente) parser do .wksp
└── execWorkflowGen.py       # (existente) executor do workflow
```

---

## 12. Fluxo Principal do Usuário

```
1. Abrir aplicação
   └─> Canvas vazio + painel de Glyphs à esquerda

2. Construir workflow (estilo Cantata)
   ├─> Clicar em "vglLoadImage" no painel → Glyph aparece no canvas
   ├─> Clicar no botão Expand (📋) do Glyph → campos de parâmetro aparecem inline
   ├─> Preencher "filename" com caminho da imagem
   ├─> Clicar em "vglCreateImage" → novo Glyph no canvas
   ├─> Clicar na porta de saída RETVAL do Glyph 1 → linha fantasma
   ├─> Clicar na porta de entrada img do Glyph 2 → conexão vermelha criada
   └─> Repetir para demais funções

3. Salvar
   └─> Ctrl+S → arquivo .wksp gerado

4. Executar
   ├─> F5 → execução inicia
   ├─> Glyphs ficam cinza (READY) → preto (RUNNING) → branco (DONE)
   ├─> Log em tempo real no painel inferior
   └─> Em caso de erro: Glyph fica vermelho, log mostra mensagem

5. Visualizar resultado
   └─> Glyph "ShowImage" após DONE → preview da imagem gerada
```

---

## 13. Fora de Escopo (v1)

- Editor de código OpenCL/CL inline
- Versionamento de workflows (git integration)
- Execução remota (apenas localhost)
- Criação de novas funções VGL pela interface
- Suporte a sub-workflows / procedures aninhados (além do básico)
- Mobile

---

## 14. Critérios de Aceite

### Canvas
- [ ] Glyphs podem ser adicionados via painel esquerdo
- [ ] Glyphs podem ser movidos livremente no canvas
- [ ] Pan e zoom funcionam
- [ ] Glyph expandido exibe campos de parâmetro editáveis inline
- [ ] Botão Delete remove o Glyph e suas conexões

### Conexões
- [ ] Clicar porta saída → clicar porta entrada cria conexão vermelha
- [ ] Conexões inválidas são rejeitadas
- [ ] Deletar conexão com tecla Delete

### Execução e Feedback Visual
- [ ] F5 executa o workflow
- [ ] Glyphs mudam de cor: cinza → preto → branco conforme execução
- [ ] Log em tempo real no painel inferior
- [ ] Glyph vermelho em caso de erro

### Persistência
- [ ] Salvar gera `.wksp` válido para o interpretador existente
- [ ] Abrir `.wksp` reconstrói o grafo visual corretamente
