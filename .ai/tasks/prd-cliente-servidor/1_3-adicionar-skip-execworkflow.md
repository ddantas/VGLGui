---
id: 1.3
fase: 1
titulo: Adicionar _check_skip() em execWorkflow.py
requisitos: [RF-006]
depende: —
paralela: true
---

# 1.3 — Adicionar `_check_skip()` em `execWorkflow.py`

## Contexto

Para permitir pular um glyph específico durante a execução (RF-006), o executor precisa
checar um sinal externo entre glyphs. O sinal é um arquivo temporário criado pelo servidor
em `/tmp/vgl_skip_{job_id}_{glyph_id}`. A variável de ambiente `VGL_JOB_ID` é injetada
pelo servidor ao spawnar o subprocess.

A mudança é mínima: 1 função + 3 linhas no loop principal.

Ver TechSpec §3.2 para o design.

## Arquivos

- **Modificar:** `execWorkflow.py`
- **NÃO tocar:** nenhum outro arquivo

## O que implementar

### 1. Adicionar `_check_skip()` após os imports

Localizar o bloco de imports no topo de `execWorkflow.py` e adicionar a função
logo abaixo (antes de `imshow`):

```python
def _check_skip(glyph_id: str) -> bool:
    """Retorna True se o glyph deve ser pulado por sinal externo do servidor."""
    job_id = os.environ.get("VGL_JOB_ID", "")
    if not job_id:
        return False
    sig = f"/tmp/vgl_skip_{job_id}_{glyph_id}"
    if os.path.exists(sig):
        print(f"[SKIP] Glyph {glyph_id} pulado por sinal externo")
        try:
            os.unlink(sig)
        except OSError:
            pass
        return True
    return False
```

### 2. Adicionar checagem no loop principal de `execWorkflow()`

Localizar o loop em `execWorkflow()`:

```python
for vGlyph in workspace.lstGlyph:
    if vGlyph.glyph_id in processed_workflows:
        continue
```

Adicionar logo após o `continue` existente:

```python
    if _check_skip(vGlyph.glyph_id):
        GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)
        continue
```

**Resultado esperado:**

```python
for vGlyph in workspace.lstGlyph:
    if vGlyph.glyph_id in processed_workflows:
        continue
    if _check_skip(vGlyph.glyph_id):          # ← NOVO
        GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)  # ← NOVO
        continue                               # ← NOVO
    # ... resto do código sem alteração
```

## Acceptance Criteria

- AC-A: Com `VGL_JOB_ID` não definido, `_check_skip()` retorna `False` sem efeito colateral
- AC-B: Com arquivo de sinal presente, `_check_skip()` retorna `True` e deleta o arquivo
- AC-C: Glyph com sinal de skip imprime `[SKIP] Glyph {id} pulado por sinal externo`
- AC-D: Workflow sem `VGL_JOB_ID` executa normalmente (sem regressão)
- AC-E: `git diff --name-only` mostra apenas `execWorkflow.py`

## Verificar

```bash
cd /home/joao/Documents/TCC_1/VGLGui
source my_env/bin/activate

# Teste isolado da função
python -c "
import os
os.environ['VGL_JOB_ID'] = 'test-job'
open('/tmp/vgl_skip_test-job_g1', 'w').close()

# importa só a função (sem rodar o módulo inteiro)
import importlib.util, sys
# Verifica que o arquivo foi criado
assert os.path.exists('/tmp/vgl_skip_test-job_g1')
print('arquivo de sinal criado OK')
"

git diff --name-only   # deve mostrar só execWorkflow.py
```
