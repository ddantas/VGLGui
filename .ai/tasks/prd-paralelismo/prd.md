---
id: prd-paralelismo
title: Paralelismo de Ramos Independentes em Workflows VGLGui
status: draft
created: 2026-03-16
classification: Medium
---

# PRD — Paralelismo de Ramos Independentes em Workflows VGLGui

## 1. Resumo Executivo

O executor atual (`execWorkflow.py`) processa todos os nós de um workflow sequencialmente, mesmo quando existem ramos sem dependência entre si. Este PRD especifica a implementação de execução paralela de ramos independentes do DAG, com feedback visual em tempo real na GUI. A feature reduz o tempo total de execução em workflows com múltiplos ramos (ex.: `fundus.wksp`, `fundus_cv.wksp`) e torna o comportamento paralelo observável pelo usuário via colorização de nós. Baseia-se nos conceitos de *Activities Parallelism* e *Pipeline Parallelism* descritos em Watanabe & Braghetto (IEEE SCC 2018).

---

## 2. Contexto e Motivação

- Workflows como `fundus.wksp` e `fundus_cv.wksp` possuem ramos completamente independentes após o nó de conversão RGB→Gray (ex.: ramo de convolução e ramo de dilatação/erosão), que hoje rodam um após o outro.
- O executor é chamado por `runner.py` em background thread — a GUI já está preparada para execução assíncrona.
- O formato `.wksp` e o parser `readWorkflow.py` já constroem uma estrutura de grafo (`workspace`) com `glyph_id` e `NodeConnection` que permite derivar o DAG de dependências em runtime.
- Motivação acadêmica: o TCC precisa demonstrar benefícios de paralelismo em workflows de processamento de imagem, comparável ao que o paper de referência demonstrou para workflows data-intensive.

---

## 3. Escopo e Non-Goals

### Em escopo (v1)

- Detecção automática de ramos independentes a partir do DAG em runtime
- Execução paralela de nós sem dependência entre si (VGL_CL e VGL_CV)
- Colorização de nós no canvas para indicar: aguardando / executando / concluído / erro
- Destaque visual dos ramos paralelos identificados (antes da execução)
- Destaque do caminho crítico (sequência de nós com maior tempo acumulado)

### Non-Goals

| ID | Non-Goal | Motivo |
|----|----------|--------|
| NG-001 | Paralelismo a nível de kernel GPU (dentro de um único nó VGL_CL) | O OpenCL já gerencia isso internamente |
| NG-002 | Execução distribuída em múltiplas máquinas | Fora do escopo do TCC |
| NG-003 | Load balancing dinâmico entre nós | Complexidade desnecessária para a v1 |
| NG-004 | Anotações explícitas do usuário sobre paralelismo | Deve ser automático, sem intervenção |
| NG-005 | Alteração do formato `.wksp` | Arquivos existentes devem continuar funcionando |
| NG-006 | Paralelismo dentro de Procedures | Procedures são tratadas como nó atômico na v1 |
| NG-007 | Métricas de tempo por nó persistidas em arquivo | Log no painel de execução é suficiente |

---

## 4. Requisitos Funcionais

---

### RF-001 [MUST]: Análise de Dependências do DAG

O sistema DEVE, ao iniciar a execução, construir um grafo de dependências a partir das `NodeConnection` do workspace e derivar grupos de nós que podem ser executados em paralelo (níveis topológicos).

**Acceptance Criteria:**
- AC-001-A: Dado um workflow com dois ramos independentes (A→B e A→C sem conexão entre B e C), B e C são identificados como pertencentes ao mesmo nível de execução paralela.
- AC-001-B: Dado um workflow linear (A→B→C), nenhum par de nós é identificado como paralelo.
- AC-001-C: O grafo é derivado exclusivamente das `NodeConnection` existentes no arquivo `.wksp`, sem necessidade de anotação manual.

---

### RF-002 [MUST]: Execução Paralela de Nós Independentes

O sistema DEVE executar nós do mesmo nível topológico em threads simultâneas, aguardando a conclusão de todos antes de avançar ao próximo nível.

**Acceptance Criteria:**
- AC-002-A: Em workflow com 2 ramos paralelos de comprimento N, o tempo de execução é ≤ tempo do ramo mais lento + overhead de sincronização (não a soma dos dois ramos).
- AC-002-B: O resultado final (imagens salvas) é idêntico ao da execução serial para o mesmo workflow e mesmos inputs.
- AC-002-C: Se um nó lança exceção em sua thread, a execução dos demais nós do mesmo nível é aguardada antes de propagar o erro; o log registra qual nó falhou.
- AC-002-D: O sistema suporta nós VGL_CL e VGL_CV no mesmo nível paralelo.

---

### RF-003 [MUST]: Progresso por Nó Durante Execução

O sistema DEVE atualizar a cor/aparência de cada nó no canvas conforme seu estado de execução.

**Acceptance Criteria:**
- AC-003-A: Nó no estado *aguardando* exibe cor padrão (sem mudança).
- AC-003-B: Nó no estado *executando* exibe cor amarela/laranja no canvas.
- AC-003-C: Nó no estado *concluído* exibe cor verde no canvas.
- AC-003-D: Nó no estado *erro* exibe cor vermelha no canvas.
- AC-003-E: As atualizações de cor são visíveis durante a execução (não apenas ao final).
- AC-003-F: Ao iniciar nova execução, todos os nós retornam ao estado/cor padrão.

---

### RF-004 [SHOULD]: Indicação Visual de Ramos Paralelos

O sistema DEVE destacar visualmente, antes da execução, quais nós pertencem ao mesmo nível de execução paralela.

**Acceptance Criteria:**
- AC-004-A: Nós do mesmo nível paralelo compartilham a mesma cor de borda ou marcação visual distinta dos demais níveis.
- AC-004-B: O destaque é gerado automaticamente ao carregar ou modificar o workflow, sem ação do usuário.
- AC-004-C: Workflows com zero paralelismo (lineares) não exibem nenhum destaque especial.

---

### RF-005 [COULD]: Caminho Crítico Destacado

O sistema DEVE identificar e destacar o caminho crítico do DAG (sequência de nós com maior número de dependências encadeadas).

**Acceptance Criteria:**
- AC-005-A: Os nós pertencentes ao caminho crítico são visualmente distinguíveis dos demais (ex.: borda mais espessa ou ícone).
- AC-005-B: Em workflows lineares, o caminho crítico é o workflow inteiro.
- AC-005-C: O caminho crítico é recalculado ao carregar um novo workflow.

---

## 5. Requisitos Não-Funcionais

| ID | Requisito | Threshold |
|----|-----------|-----------|
| RNF-001 | Redução de tempo de execução | ≥ 20% em workflows com ≥ 2 ramos paralelos de comprimento similar |
| RNF-002 | Corretude | Resultado binariamente idêntico ao serial (mesmos pixels nas imagens de saída) |
| RNF-003 | Responsividade da GUI | GUI permanece interativa durante execução paralela (sem freeze do event loop) |
| RNF-004 | Fallback serial | Em caso de erro de threading, o sistema cai em modo serial sem perda de dados |
| RNF-005 | Overhead de análise do DAG | A análise de dependências adiciona < 100ms ao início da execução |
| RNF-006 | Atualização de cor dos nós | Mudanças de estado visíveis em ≤ 500ms após a transição real do nó |

---

## 6. Critérios de Sucesso

| Métrica | Baseline | Target | Método de Medição |
|---------|----------|--------|-------------------|
| Tempo de execução do `fundus_cv.wksp` | T_serial (medido antes da feature) | T_serial × 0.80 ou menor | `time` no log de execução |
| Corretude | — | 100% idêntico ao serial | diff binário das imagens de saída |
| Crash durante execução paralela | — | 0 crashes em 10 execuções consecutivas | execução manual |
| Feedback visual de nós | Nenhum | Todos os estados (aguardando/executando/concluído/erro) visíveis | inspeção visual |

---

## 7. Riscos e Dependências

| ID | Risco | Probabilidade | Impacto | Mitigação |
|----|-------|---------------|---------|-----------|
| R-001 | Contexto OpenCL não é thread-safe — dois nós VGL_CL em paralelo podem travar | Alta | Alto | Usar lock global para nós VGL_CL; apenas nós VGL_CV rodam em paralelo real na v1 se confirmado |
| R-002 | GIL do Python limita paralelismo real em threads CPU-bound | Média | Médio | Avaliar uso de `ProcessPoolExecutor` para VGL_CV; threads para I/O e sincronização |
| R-003 | Atualização de widgets DPG fora da thread principal pode causar crash | Alta | Alto | Usar fila de mensagens (já existe padrão com `flush_log_buffer` em `app.py`) |
| R-004 | Workflows com procedures — grafo interno não está exposto ao executor principal | Baixa | Baixo | Procedures tratadas como nó atômico (NG-006) |

**Dependências:**
- `execWorkflow.py` — arquivo principal a modificar
- `runner.py` — orquestra a execução em background thread; precisa repassar sinais de progresso
- `canvas.py` — precisa expor API para colorir nós por `glyph_id`
- `gui/app.py` — fila de atualizações de UI já existe (`flush_status_queue`, `flush_log_buffer`); o mesmo padrão deve ser usado para atualização de cor dos nós

---

## 8. Proteções — NÃO ALTERAR

- **Formato `.wksp`**: nenhuma linha nova ou campo novo no formato de arquivo
- **`readWorkflow.py`**: parser não deve ser modificado
- **`glyph_registry.py`**: definições de portas e parâmetros não mudam
- **API pública de `runner.py`**: `start_run()` e `stop_run()` mantêm a mesma assinatura
- **Modo serial como fallback**: a execução serial deve continuar funcionando como default seguro

---

## 9. Questões em Aberto

| ID | Questão | Owner | Default |
|----|---------|-------|---------|
| Q-001 | O contexto OpenCL (`vglContext`) é thread-safe para leituras simultâneas? | Implementador | Assumir não-seguro; serializar VGL_CL com lock |
| Q-002 | `ProcessPoolExecutor` ou `ThreadPoolExecutor` para VGL_CV? | Implementador | ThreadPoolExecutor (mais simples; GIL liberado por operações de I/O e C extensions do OpenCV) |
| Q-003 | A colorização de nós via DPG deve usar `node_attribute` theme ou `dpg.bind_item_theme`? | Implementador | `dpg.bind_item_theme` por `glyph_id` |
