---
id: 2.2
fase: 2
titulo: Verificar regressão — execução serial produz output idêntico
requisitos: [RF-002]
depende: 1.1
paralela: true  # pode rodar em paralelo com 2.1
---

# 2.2 — Verificar regressão serial

Confirma que a mudança em `opencl_context.py` não alterou o comportamento
da execução serial (modo atual de uso). O resultado das imagens geradas
deve ser binariamente idêntico ao baseline pré-mudança.

## Arquivos

- **Ler:** `SAMPLES/demo.wksp` — workflow de referência para regressão
- **Não criar nem modificar arquivos de código**

## Passos de verificação

### 1. Capturar baseline (ANTES de aplicar task 1.1)

Se task 1.1 ainda não foi aplicada, pular este passo (baseline já deve existir).
Se a mudança já foi feita e não há baseline, usar `git stash` para reverter
temporariamente, gerar o baseline, e depois `git stash pop`.

```bash
cd /home/joao/Documents/TCC_1/VGLGui
source my_env/bin/activate

# gerar saída baseline
python3 execWorkflow.py SAMPLES/demo.wksp

# guardar cópias das saídas
cp out/conv.png /tmp/baseline_conv.png
cp out/dilate.png /tmp/baseline_dilate.png
```

### 2. Rodar com a mudança aplicada

```bash
python3 execWorkflow.py SAMPLES/demo.wksp
```

### 3. Comparar binariamente

```bash
diff /tmp/baseline_conv.png out/conv.png    && echo "conv.png: OK"
diff /tmp/baseline_dilate.png out/dilate.png && echo "dilate.png: OK"
```

## Acceptance Criteria

- AC-A: `diff /tmp/baseline_conv.png out/conv.png` retorna exit code 0 (sem diferença)
- AC-B: `diff /tmp/baseline_dilate.png out/dilate.png` retorna exit code 0
- AC-C: Execução de `execWorkflow.py` não lança exceção

## Verificar

```bash
diff /tmp/baseline_conv.png out/conv.png && diff /tmp/baseline_dilate.png out/dilate.png
echo "Regressão: OK"
```
