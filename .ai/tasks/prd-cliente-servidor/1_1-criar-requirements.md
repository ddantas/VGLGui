---
id: 1.1
fase: 1
titulo: Criar requirements.txt com dependências novas
requisitos: [RF-001, RNF-003]
depende: —
paralela: true
---

# 1.1 — Criar `requirements.txt`

## Contexto

O projeto não tem `requirements.txt`. O servidor FastAPI e o cliente HTTP/WS precisam
de 4 novas dependências. Todas devem ser instaladas no venv `my_env/`.

## Arquivos

- **Criar:** `requirements.txt` (raiz do projeto)
- **NÃO tocar:** nenhum outro arquivo

## O que fazer

### 1. Verificar o que já está instalado no venv

```bash
source my_env/bin/activate
pip show fastapi uvicorn httpx websockets 2>&1
```

### 2. Criar `requirements.txt`

```
# Servidor de execução
fastapi>=0.111.0
uvicorn[standard]>=0.29.0

# Cliente HTTP/WS (runner.py)
httpx>=0.27.0
websockets>=12.0
```

### 3. Instalar as que faltarem

```bash
source my_env/bin/activate
pip install -r requirements.txt
```

## Acceptance Criteria

- AC-A: `python -c "import fastapi, uvicorn, httpx, websockets; print('OK')"` não lança exceção
- AC-B: `requirements.txt` existe na raiz do projeto com as 4 dependências
- AC-C: Nenhum outro arquivo do projeto foi alterado

## Verificar

```bash
cd /home/joao/Documents/TCC_1/VGLGui
source my_env/bin/activate
python -c "import fastapi, uvicorn, httpx, websockets; print('OK')"
git diff --name-only   # não deve mostrar nada (requirements.txt é novo/untracked)
```
