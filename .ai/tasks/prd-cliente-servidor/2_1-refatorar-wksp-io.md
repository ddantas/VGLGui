---
id: 2.1
fase: 2
titulo: Refatorar gui/wksp_io.py — extrair save_wksp_to_string()
requisitos: [RF-007]
depende: 1.1
paralela: false
---

# 2.1 — Refatorar `gui/wksp_io.py`

## Contexto

O `runner.py` refatorado (task 2.2) precisa enviar o conteúdo do workspace como string
para o servidor via `POST /run`. Atualmente `save_wksp()` só grava em arquivo.

A solução é extrair a lógica de escrita para `_write_wksp(f)` que aceita qualquer
file-like object, e adicionar `save_wksp_to_string()` que usa `io.StringIO`.

## Arquivos

- **Modificar:** `gui/wksp_io.py`
- **NÃO tocar:** nenhum outro arquivo

## O que implementar

### 1. Ler `gui/wksp_io.py` completo antes de editar

### 2. Extrair corpo de `save_wksp()` para `_write_wksp(f)`

**Antes (estrutura atual):**
```python
def save_wksp(path: str):
    with open(path, "w") as f:
        # ... todo o código de escrita aqui ...
```

**Depois:**
```python
def _write_wksp(f):
    """Escreve o workspace no file-like object f."""
    # ... todo o código de escrita (movido de save_wksp) ...

def save_wksp(path: str):
    with open(path, "w") as f:
        _write_wksp(f)

def save_wksp_to_string() -> str:
    """Serializa o workspace atual para string sem gravar arquivo."""
    import io
    buf = io.StringIO()
    _write_wksp(buf)
    return buf.getvalue()
```

### 3. Garantir que `save_wksp()` continua funcionando identicamente

O comportamento de `save_wksp(path)` não muda — apenas delega para `_write_wksp`.

## Acceptance Criteria

- AC-A: `save_wksp(path)` continua gravando arquivo com conteúdo correto
- AC-B: `save_wksp_to_string()` retorna string com mesmo conteúdo que `save_wksp()` gravaria
- AC-C: `save_wksp_to_string()` não cria nenhum arquivo no disco
- AC-D: `git diff --name-only` mostra apenas `gui/wksp_io.py`

## Verificar

```bash
cd /home/joao/Documents/TCC_1/VGLGui
source my_env/bin/activate
git diff --name-only   # deve mostrar só gui/wksp_io.py
```

Teste rápido (com a GUI rodando e um workflow carregado):
```python
from gui.wksp_io import save_wksp, save_wksp_to_string
import tempfile, os

# Salva para arquivo
tmp = tempfile.mktemp(suffix=".wksp")
save_wksp(tmp)
file_content = open(tmp).read()
os.unlink(tmp)

# Serializa para string
str_content = save_wksp_to_string()

assert file_content == str_content, "FALHOU: conteúdos diferentes"
print("OK: save_wksp_to_string() equivalente ao save_wksp()")
```
