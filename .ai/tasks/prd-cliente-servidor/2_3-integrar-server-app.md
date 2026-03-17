---
id: 2.3
fase: 2
titulo: Integrar start/stop do servidor em gui/app.py
requisitos: [RF-008, RNF-005]
depende: 1.2, 2.2
paralela: false
---

# 2.3 — Integrar start/stop do servidor em `gui/app.py`

## Contexto

O servidor deve subir automaticamente quando a GUI abre e ser encerrado quando
a GUI fecha. Se não responder em 5 segundos, a GUI registra o erro no log e
continua funcionando (mas o botão Run fica desabilitado).

Ver TechSpec §3.4 para o design.

## Arquivos

- **Modificar:** `gui/app.py`
- **NÃO tocar:** nenhum outro arquivo

## O que implementar

### 1. Adicionar imports no topo de `gui/app.py`

```python
import atexit as _atexit
import subprocess as _sp
import time as _time

import httpx as _httpx
```

### 2. Adicionar variável e funções de ciclo de vida

Adicionar após os imports, antes de `run()`:

```python
_server_proc: "_sp.Popen | None" = None


def _start_server() -> bool:
    """Sobe executor_server.py e aguarda /health. Retorna True se OK."""
    global _server_proc
    server_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "executor_server.py"
    )
    _server_proc = _sp.Popen(
        [sys.executable, server_script, "8765"],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    _atexit.register(_stop_server)

    for _ in range(50):                          # tenta por 5 segundos
        _time.sleep(0.1)
        try:
            r = _httpx.get("http://127.0.0.1:8765/health", timeout=0.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


def _stop_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=3)
        except Exception:
            _server_proc.kill()
```

### 3. Chamar `_start_server()` no início de `run()`

Localizar a função `run()` e adicionar logo após `set_language(...)`:

```python
def run():
    _cfg = load_config()
    set_language(_cfg.get("language", "pt"))

    # ── NOVO ──────────────────────────────────────────────────────────────
    if not _start_server():
        APP_STATE["server_ok"] = False
        # O log ainda não existe aqui — imprime no stdout
        print("[ERRO] Servidor de execução não respondeu na porta 8765.")
    else:
        APP_STATE["server_ok"] = True
    # ── FIM NOVO ──────────────────────────────────────────────────────────

    # ... resto de run() sem alteração (dpg.create_context etc.)
```

### 4. Adicionar `"server_ok"` ao APP_STATE

Localizar o dict `APP_STATE` em `app.py` e adicionar a chave:

```python
APP_STATE: dict = {
    ...
    "server_ok": False,    # ← NOVO: True após servidor responder
    ...
}
```

### 5. (Opcional) Desabilitar botão Run se servidor não está OK

No `toolbar.py`, o botão Run chama `run_workflow()`. Adicionar guarda:

```python
# Em run_workflow() — já existe esta verificação no runner.py
# Não precisa mudar toolbar.py; o runner.py já loga o erro
```

## Acceptance Criteria

- AC-A: GUI abre e `APP_STATE["server_ok"] == True` após `_start_server()`
- AC-B: Ao fechar a GUI, o processo `executor_server.py` é terminado (verificar com `ps aux`)
- AC-C: Se porta 8765 ocupada ou servidor falhar, `APP_STATE["server_ok"] == False` e GUI continua abrindo
- AC-D: `git diff --name-only` mostra apenas `gui/app.py`

## Verificar

```bash
# Antes de abrir a GUI
ps aux | grep executor_server   # não deve aparecer nada

# Abrir a GUI
python3 gui/main.py &
sleep 3
ps aux | grep executor_server   # deve aparecer o processo

# Fechar a GUI (Ctrl+C ou fechar janela)
sleep 1
ps aux | grep executor_server   # não deve aparecer mais

git diff --name-only   # deve mostrar só gui/app.py
```
