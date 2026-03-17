---
id: 3.1
fase: 3
titulo: Verificação end-to-end
requisitos: [todos]
depende: 2.3
paralela: false
---

# 3.1 — Verificação End-to-End

## Contexto

Verificação completa de todos os requisitos do PRD após a implementação das fases 1 e 2.
Executa os critérios de verificação definidos na TechSpec §6.

## Checklist de Verificação

### RF-001 — Servidor de Execução

```bash
# 1. Verificar que servidor sobe com a GUI
python3 gui/main.py &
sleep 3
curl -s http://localhost:8765/health
# Esperado: {"status":"ok"}
```

- [ ] `GET /health` retorna 200
- [ ] `APP_STATE["server_ok"] == True` (verificar no log ou adicionar print temporário)
- [ ] Servidor termina quando a GUI fecha

---

### RF-002 — Iniciar Execução

```bash
# Com a GUI aberta, carregar demo.wksp e clicar Run
# Ou testar via curl:
WKSP=$(cat exemplos/demo.wksp | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
curl -s -X POST http://localhost:8765/run \
  -H "Content-Type: application/json" \
  -d "{\"wksp_content\": $WKSP, \"device\": \"CPU\"}"
# Esperado: {"job_id":"<uuid>","status":"running"}
```

- [ ] Retorna HTTP 202 + job_id UUID
- [ ] Segundo `POST /run` com job ativo retorna HTTP 409

---

### RF-003 — Eventos WebSocket

```bash
# Conectar ao WS enquanto job roda
JOB_ID="<id do job acima>"
python3 -c "
import asyncio, websockets, json
async def main():
    async with websockets.connect('ws://localhost:8765/events/$JOB_ID') as ws:
        async for msg in ws:
            e = json.loads(msg)
            print(e['type'], e.get('glyph_id',''), e.get('line','')[:60])
            if e['type'] == 'finished':
                break
asyncio.run(main())
"
```

- [ ] Eventos `glyph_start` aparecem no WS
- [ ] Eventos `glyph_done` aparecem no WS
- [ ] Evento `finished` aparece ao final
- [ ] Conectar após job iniciado recebe replay dos eventos anteriores

---

### RF-004 — Status

```bash
curl -s http://localhost:8765/status/$JOB_ID | python3 -m json.tool
```

- [ ] Retorna `status`, `glyphs`, `current_glyph`
- [ ] Retorna 404 para job_id inexistente

---

### RF-005 — Stop Completo

```bash
# Iniciar job longo, depois parar
curl -s -X POST http://localhost:8765/stop/$JOB_ID
# Esperado: {"ok":true}
```

- [ ] Job para imediatamente
- [ ] Evento `stopped` emitido no WS
- [ ] Novo `POST /run` funciona após stop

---

### RF-006 — Skip de Glyph

```bash
# Com job rodando, criar sinal de skip para um glyph específico
curl -s -X POST http://localhost:8765/stop/$JOB_ID/glyph/g3
# Verificar que arquivo foi criado:
ls /tmp/vgl_skip_${JOB_ID}_g3
# Verificar no log da GUI que aparece [SKIP] Glyph g3 pulado
```

- [ ] Arquivo de sinal criado
- [ ] Log mostra `[SKIP]` para o glyph
- [ ] Execução continua para próximo glyph

---

### RF-007 — Runner sem regressão

- [ ] `demo.wksp` executa e produz as mesmas imagens de saída que antes
- [ ] `fundus.wksp` executa e produz as mesmas imagens de saída que antes
- [ ] Canvas coloriza glyphs (cinza → azul → verde) corretamente

---

### RF-008 — Ciclo de vida do servidor

```bash
ps aux | grep executor_server   # antes de abrir GUI: nada
python3 gui/main.py &
sleep 3
ps aux | grep executor_server   # deve aparecer
# fechar GUI
sleep 1
ps aux | grep executor_server   # não deve aparecer
```

- [ ] Servidor sobe automaticamente com a GUI
- [ ] Servidor termina quando GUI fecha

---

### RNF-002 — Sem regressão de performance

```bash
# Medir tempo antes (com runner.py antigo via git stash, ou baseline manual)
# Medir tempo depois
# Diferença deve ser < 5%
```

- [ ] Tempo de `fundus.wksp` não aumentou mais de 5%

---

### RNF-004 — Arquivos .wksp existentes

- [ ] `exemplos/demo.wksp` carrega e executa sem erro
- [ ] `exemplos/fundus.wksp` (se existir) carrega e executa sem erro
- [ ] Arquivo `.wksp` salvo pela GUI abre normalmente

---

## Resultado

Preencher após verificação:

| Requisito | Status | Observação |
|-----------|--------|------------|
| RF-001 | ⬜ | |
| RF-002 | ⬜ | |
| RF-003 | ⬜ | |
| RF-004 | ⬜ | |
| RF-005 | ⬜ | |
| RF-006 | ⬜ | |
| RF-007 | ⬜ | |
| RF-008 | ⬜ | |
| RNF-002 | ⬜ | |
| RNF-004 | ⬜ | |
