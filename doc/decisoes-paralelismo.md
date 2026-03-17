# Decisões Técnicas — Paralelismo de Atividades no VGLGui

**Referência:** Watanabe & Braghetto, "Improving Parallelism in Data-Intensive Workflows with Distributed Databases", IEEE SCC 2018, DOI 10.1109/SCC.2018.00034.

---

## Contexto e motivação

O artigo trata de workflows modelados como DAGs (Directed Acyclic Graphs), onde nós representam atividades de processamento e arestas representam fluxos de dados entre elas. O VGLGui segue exatamente esse modelo: cada Glyph é um nó e cada `NodeConnection` é uma aresta dirigida.

O principal argumento do artigo é: **o makespan (tempo total de execução) de um workflow é determinado pelo caminho crítico do DAG**, não pela soma de todas as atividades. Atividades que não pertencem ao caminho crítico e não têm dependências entre si desperdiçam tempo de execução ao serem processadas serialmente.

Hoje, o executor VGLGui (`execWorkflow.py`) percorre o DAG em ordem topológica e executa **cada nó serialmente**, mesmo que múltiplos nós sejam independentes entre si.

---

## As três estratégias do artigo e o que se aplica

### 1. Activities Parallelism ✅ — IMPLEMENTAR

**Definição:** atividades sem dependência de dados (direta ou indireta) são executadas simultaneamente em recursos computacionais distintos.

**Por que aplicar ao VGLGui:**

Em um workflow de processamento de imagens, é comum que após uma operação de carga (`vglLoadImage`) o grafo se divida em múltiplos branches independentes — por exemplo, aplicar convolução e dilatação sobre a mesma imagem de entrada. Sem paralelismo, esses branches rodam sequencialmente. Com paralelismo por atividade, cada branch ocupa um thread distinto da CPU e submete comandos para a GPU via sua própria `cl.CommandQueue`.

O ganho teórico é proporcional à largura máxima do DAG no nível de maior concorrência. Para um workflow com `k` branches independentes no nível mais largo, o speedup teórico se aproxima de `k` (limitado por Amdahl pelo overhead serial de load/save).

**Viabilidade técnica:**
- O VGLGui já parseia o DAG completo antes de executar (`readWorkflow.py`)
- A análise de dependências pode ser feita por ordenação topológica + cálculo de níveis de execução
- A thread-local `cl.CommandQueue` (PR #9) garante que threads paralelas submetam comandos à GPU sem condição de corrida
- `cl.Context` é thread-safe para criação de filas e buffers; apenas `cl.CommandQueue` não era — já corrigido

**Complexidade de implementação:** baixa. O executor precisa:
1. Calcular o nível topológico de cada nó
2. Agrupar nós do mesmo nível que não compartilham dados de saída
3. Executar cada grupo com `concurrent.futures.ThreadPoolExecutor`
4. Chamar `queue.finish()` ao término de cada grupo antes de avançar

---

### 2. Pipeline Parallelism ⚠️ — NÃO APLICAR agora

**Definição:** atividades sequenciais se sobrepõem — a atividade `i+1` começa a processar dados parciais produzidos por `i` antes de `i` terminar completamente.

**Por que não aplicar:** operações OpenCL sobre imagens são atômicas no sentido prático — a saída de um kernel só é válida após `queue.finish()`. Não há produção incremental de subpixels. Implementar pipeline parallelism exigiria particionar a imagem em tiles, executar o kernel por tile, e sincronizar tile a tile — complexidade significativa com ganho marginal dado o overhead de transferência.

---

### 3. Data Parallelism ❌ — FORA DO ESCOPO

**Definição:** a mesma atividade roda `n` réplicas, cada uma processando um subconjunto disjunto dos dados de entrada.

**Por que não aplicar:** o VGLGui processa uma imagem por execução de workflow. Dados parallelism faz sentido para processamento em batch (múltiplas imagens). Não está no escopo atual.

---

## Análise de dependências no DAG

Para identificar quais nós podem rodar em paralelo, calculamos o **nível topológico** de cada nó:

```
nível[v] = 0                              se v não tem predecessores
nível[v] = max(nível[u] + 1)             para todo u → v
```

Nós com o mesmo nível e sem aresta entre si são candidatos a execução paralela. Adicionalmente, precisamos verificar se os nós não leem/escrevem a **mesma imagem OpenCL** (mesmo `VglImage` objeto) — colisão de dados que não aparece nas arestas do `.wksp` se as operações forem in-place.

Exemplo para `demo.wksp`:

```
Nível 0: vglLoadImage(input)
Nível 1: vglConvolution, vglDilate          ← PARALELO (saídas distintas)
Nível 2: vglSaveImage(conv), vglSaveImage(dilate)  ← PARALELO
```

Makespan serial:   T_load + T_conv + T_dilate + T_save1 + T_save2
Makespan paralelo: T_load + max(T_conv, T_dilate) + max(T_save1, T_save2)

---

## Ganho esperado

O artigo reporta reduções de até **66,6%** no makespan em cenários com alta concorrência. Para o VGLGui, com GPU como backend e workflows de 10–30 nós com branches de profundidade 2–4, a estimativa conservadora é de **20–40% de redução** no tempo de execução em workflows com múltiplas branches independentes (`fundus.wksp`, `fundus_cv.wksp`).

---

## Plano de implementação

| Fase | O que fazer | Status |
|---|---|---|
| 1 | Thread-local `cl.CommandQueue` em `opencl_context.py` | ✅ PR #9 |
| 2 | Análise de níveis topológicos no DAG do executor | pendente |
| 3 | `ThreadPoolExecutor` por nível no `execWorkflow.py` | pendente |
| 4 | Sincronização entre níveis com `finish_queue()` | pendente |
| 5 | Testes de regressão + benchmark serial vs paralelo | pendente |

### Restrições de implementação
- Procedures devem ser tratadas como nós atômicos (execução interna permanece serial por enquanto)
- Nós que operam in-place na mesma imagem não podem ser paralelizados mesmo sem aresta entre eles — o grafo de dependências deve ser enriquecido com informação de dados compartilhados
- Manter execução serial como modo padrão configurável (`config.json`) para garantir reprodutibilidade de resultados
