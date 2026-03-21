#!/usr/bin/env python3
"""
executor_server.py — Servidor de execução de workflows VGLGui (Camada 3 / Interpretation)
Referência arquitetural: Maciel 2021, Fig. 6, Camadas 1-3
"""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
_FUNC_START_RE = re.compile(r"A função (\S+) está sendo executada")
_SHOW_IMG_RE   = re.compile(r"\[GUI_SHOW\] (.+)")
_SKIP_RE       = re.compile(r"\[SKIP\] Glyph (\S+) pulado")
_PAUSED_RE     = re.compile(r"\[PAUSED\] Glyph (\S+) pausado")
_RESUMED_RE    = re.compile(r"\[RESUMED\] Glyph (\S+) retomado")

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    wksp_content: str
    device: str = "GPU"
    breakpoints: list = []   # glyph_ids com breakpoint pré-definido


@dataclass
class JobState:
    job_id: str
    device: str
    status: str                               # "running"|"done"|"error"|"stopped"|"paused"
    process: object                           # subprocess.Popen | None
    events: list = field(default_factory=list)
    ws_queues: list = field(default_factory=list)
    glyph_status: dict = field(default_factory=dict)
    glyph_list: list = field(default_factory=list)  # [{"id": str, "func": str}, ...]
    breakpoints: set = field(default_factory=set)
    current_glyph: Optional[str] = None
    wksp_tmp: str = ""
    _t0: float = 0.0


# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
_job: Optional[JobState] = None
_job_lock = asyncio.Lock()
_func_map: dict = {}   # func_name → [glyph_id, ...]

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="VGLGui Executor Server")


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_env(device: str) -> dict:
    env = os.environ.copy()
    if device == "CPU":
        env["LD_LIBRARY_PATH"] = (
            "/opt/AMDAPPSDK-2.9-1/lib/x86_64/:" + env.get("LD_LIBRARY_PATH", "")
        )
    else:
        env["LD_LIBRARY_PATH"] = (
            "/opt/amdgpu/lib/x86_64-linux-gnu/:/opt/rocm/lib/:"
            + env.get("LD_LIBRARY_PATH", "")
        )
    return env


def _skip_path(job_id: str, glyph_id: str) -> str:
    return f"/tmp/vgl_skip_{job_id}_{glyph_id}"


def _break_path(job_id: str, glyph_id: str) -> str:
    return f"/tmp/vgl_break_{job_id}_{glyph_id}"


def _continue_path(job_id: str) -> str:
    return f"/tmp/vgl_continue_{job_id}"


def _cleanup_tmp(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


async def _emit(job: JobState, event: dict):
    """Adiciona timestamp, guarda no buffer, distribui para clientes WS."""
    event.setdefault("timestamp", _now())
    job.events.append(event)

    if event["type"] == "glyph_start":
        job.glyph_status[event["glyph_id"]] = "running"
        job.current_glyph = event["glyph_id"]
    elif event["type"] == "glyph_done":
        job.glyph_status[event["glyph_id"]] = event["status"]
    elif event["type"] == "glyph_paused":
        job.glyph_status[event["glyph_id"]] = "paused"
        job.status = "paused"
    elif event["type"] == "glyph_resumed":
        job.glyph_status[event["glyph_id"]] = "running"
        job.status = "running"

    for q in list(job.ws_queues):
        await q.put(event)


def _find_glyph(func_name: str, executed: set) -> Optional[str]:
    for gid in _func_map.get(func_name, []):
        if gid not in executed:
            return gid
    return None


def _build_func_map(wksp_content: str) -> list:
    """Parseia o .wksp e constrói func_name → [glyph_ids].
    Suporta dois formatos de linha Glyph:
      Novo (GUI):  Glyph:library:func::localhost:id:X:Y::  (parts[3] == "")
      Antigo:      Glyph:library:func:localhost:id:X:Y:    (parts[3] != "")
    Retorna lista ordenada de {"id": gid, "func": func}.
    """
    global _func_map
    _func_map = {}
    glyph_list = []
    for line in wksp_content.splitlines():
        s = line.strip()
        if s.startswith("Glyph:") or s.startswith("ProcedureBegin:"):
            parts = s.split(":")
            if len(parts) >= 6:
                func = parts[2]
                gid = parts[5] if parts[3] == "" else parts[4]
                _func_map.setdefault(func, []).append(gid)
                glyph_list.append({"id": gid, "func": func})
    return glyph_list


# ---------------------------------------------------------------------------
# Job runner (asyncio task)
# ---------------------------------------------------------------------------
async def _read_stdout_lines(proc):
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        yield line.rstrip()


async def _run_job(job: JobState):
    import time as _time

    job._t0 = _time.time()
    env = _build_env(job.device)
    env["VGL_JOB_ID"] = job.job_id

    executor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "execWorkflow.py")
    try:
        proc = subprocess.Popen(
            [sys.executable, executor, job.wksp_tmp, job.device],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except FileNotFoundError as e:
        await _emit(job, {"type": "error", "message": str(e)})
        job.status = "error"
        return

    job.process = proc
    executed: set = set()

    async for line in _read_stdout_lines(proc):
        sm = _SHOW_IMG_RE.search(line)
        if sm:
            await _emit(job, {
                "type": "show_image",
                "glyph_id": job.current_glyph,
                "path": sm.group(1).strip(),
            })
            continue

        await _emit(job, {"type": "log", "line": line})

        sk = _SKIP_RE.search(line)
        if sk:
            gid = sk.group(1)
            await _emit(job, {"type": "glyph_done", "glyph_id": gid, "status": "skipped"})
            executed.add(gid)
            continue

        pm = _PAUSED_RE.search(line)
        if pm:
            gid = pm.group(1)
            func = next((g["func"] for g in job.glyph_list if g["id"] == gid), gid)
            await _emit(job, {"type": "glyph_paused", "glyph_id": gid, "func": func})
            continue

        rm = _RESUMED_RE.search(line)
        if rm:
            gid = rm.group(1)
            await _emit(job, {"type": "glyph_resumed", "glyph_id": gid})
            continue

        m = _FUNC_START_RE.search(line)
        if m:
            if job.current_glyph and job.current_glyph not in executed:
                await _emit(job, {
                    "type": "glyph_done",
                    "glyph_id": job.current_glyph,
                    "status": "done",
                })
                executed.add(job.current_glyph)

            func = m.group(1)
            gid = _find_glyph(func, executed)
            if gid:
                await _emit(job, {"type": "glyph_start", "glyph_id": gid, "func": func})

    await asyncio.get_event_loop().run_in_executor(None, proc.wait)
    rc = proc.returncode
    elapsed = _time.time() - job._t0

    if job.current_glyph and job.current_glyph not in executed:
        await _emit(job, {
            "type": "glyph_done",
            "glyph_id": job.current_glyph,
            "status": "done" if rc == 0 else "error",
        })

    if job.status in ("running", "paused"):
        job.status = "done" if rc == 0 else ("stopped" if rc == -15 else "error")

    await _emit(job, {
        "type": "finished",
        "returncode": rc,
        "elapsed": round(elapsed, 3),
    })

    _cleanup_tmp(job.wksp_tmp)
    job.process = None


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>VGLGui Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#1e1e2e;color:#cdd6f4;height:100vh;display:flex;flex-direction:column}
#header{background:#181825;padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #313244;flex-wrap:wrap}
#header h1{font-size:15px;color:#cba6f7;white-space:nowrap}
.dot-ok{color:#a6e3a1}
.dot-err{color:#f38ba8}
#job-info{font-size:11px;color:#a6adc8;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
button{background:#313244;color:#cdd6f4;border:none;padding:5px 11px;border-radius:4px;cursor:pointer;font-family:monospace;font-size:12px}
button:hover{background:#45475a}
button:disabled{opacity:.4;cursor:default}
#btn-stop{color:#f38ba8}
#btn-run{color:#a6e3a1}
label.file-label{background:#313244;color:#cdd6f4;padding:5px 11px;border-radius:4px;cursor:pointer;font-size:12px}
label.file-label:hover{background:#45475a}
select{background:#313244;color:#cdd6f4;border:none;padding:5px 8px;border-radius:4px;font-family:monospace;font-size:12px}
#pause-banner{background:#f9e2af18;border-top:1px solid #f9e2af55;border-bottom:1px solid #f9e2af55;color:#f9e2af;padding:8px 16px;display:none;align-items:center;gap:12px;font-size:13px}
#pause-banner.visible{display:flex}
#btn-continue{background:#a6e3a1;color:#1e1e2e;font-weight:bold}
#btn-continue:hover{background:#94e2d5}
#main{display:flex;flex:1;overflow:hidden}
#glyphs-panel{width:300px;padding:10px;overflow-y:auto;border-right:1px solid #313244;display:flex;flex-direction:column;gap:6px}
#glyphs-panel h2,#log-panel h2{font-size:10px;color:#6c7086;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.glyph-card{background:#181825;border:1px solid #313244;border-radius:5px;padding:7px 9px;display:flex;align-items:center;gap:7px;border-left:4px solid #45475a;transition:border-color .2s,background .2s}
.glyph-card[data-status=running]{border-left-color:#89b4fa}
.glyph-card[data-status=done]{border-left-color:#a6e3a1}
.glyph-card[data-status=error]{border-left-color:#f38ba8}
.glyph-card[data-status=paused]{border-left-color:#f9e2af;background:#f9e2af0d}
.glyph-card[data-status=skipped]{border-left-color:#fab387;opacity:.7}
.gdot{width:7px;height:7px;border-radius:50%;background:#45475a;flex-shrink:0}
.glyph-card[data-status=running] .gdot{background:#89b4fa;animation:pulse 1s infinite}
.glyph-card[data-status=done] .gdot{background:#a6e3a1}
.glyph-card[data-status=error] .gdot{background:#f38ba8}
.glyph-card[data-status=paused] .gdot{background:#f9e2af;animation:pulse 1s infinite}
.glyph-card[data-status=skipped] .gdot{background:#fab387}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.ginfo{flex:1;min-width:0}
.gfunc{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gid{font-size:10px;color:#6c7086}
.btn-bp{background:none;border:none;font-size:13px;cursor:pointer;padding:1px 3px;opacity:.4;line-height:1}
.btn-bp:hover,.btn-bp.active{opacity:1}
.gpreview{width:32px;height:32px;object-fit:cover;border-radius:3px;border:1px solid #313244;cursor:pointer;flex-shrink:0}
#log-panel{flex:1;display:flex;flex-direction:column;overflow:hidden}
#log-panel h2{padding:10px 12px 4px}
#log{flex:1;overflow-y:auto;padding:0 12px 12px;font-size:11px;line-height:1.7;color:#7f849c}
.lg{color:#7f849c}
.lg-gs{color:#89b4fa}
.lg-gd{color:#a6e3a1}
.lg-er{color:#f38ba8}
.lg-pa{color:#f9e2af;font-weight:bold}
.lg-fi{color:#cba6f7;font-weight:bold}
</style>
</head>
<body>
<div id="header">
  <h1>⚡ VGLGui</h1>
  <span id="srv-status" class="dot-ok">● servidor ok</span>
  <span id="job-info">sem job ativo</span>
  <input type="file" id="wksp-file" accept=".wksp" style="display:none" onchange="previewWksp()">
  <label class="file-label" for="wksp-file">📂 .wksp</label>
  <select id="dev-sel"><option value="CPU">CPU</option><option value="GPU">GPU</option></select>
  <button id="btn-run" onclick="runWorkflow()">▶ Run</button>
  <button id="btn-stop" onclick="stopJob()" disabled>■ Stop</button>
  <button onclick="init()" title="Reconectar ao job ativo">↺</button>
</div>
<div id="pause-banner">
  <span id="pause-msg">⏸ Pausado</span>
  <button id="btn-continue" onclick="continueJob()">▶ Continuar</button>
</div>
<div id="main">
  <div id="glyphs-panel">
    <h2>Glyphs <span id="glyph-count" style="color:#313244"></span></h2>
    <div id="glyph-cards"></div>
  </div>
  <div id="log-panel">
    <h2>Log</h2>
    <div id="log"></div>
  </div>
</div>
<script>
const SRV = `${location.protocol}//${location.host}`;
const WSB = `ws://${location.host}`;
let jobId = null, ws = null;
const breakpoints = new Set();

async function init() {
  const r = await fetch(`${SRV}/current`).catch(()=>null);
  if (!r||!r.ok){setSrvErr();return;}
  const d = await r.json();
  if (d.job_id) connectJob(d.job_id, d.glyphs||{}, d.glyph_list||[], false, d.status);
}

function setSrvErr() {
  const s = document.getElementById('srv-status');
  s.textContent = '● offline'; s.className = 'dot-err';
}

function connectJob(id, initGlyphs, initList, keepCards=false, knownStatus=null) {
  const isNew = !keepCards && id !== jobId;
  jobId = id;
  if (isNew) {
    document.getElementById('glyph-cards').innerHTML = '';
    document.getElementById('log').innerHTML = '';
    document.getElementById('glyph-count').textContent = '';
    breakpoints.clear();
    hideBanner();
  }
  // Atualiza job-info imediatamente com o status conhecido (não espera o WS fechar)
  const displayStatus = knownStatus || 'carregando…';
  document.getElementById('job-info').textContent = `job ${id.slice(0,8)}… | ${displayStatus}`;
  document.getElementById('btn-stop').disabled = false;
  if (initList && initList.length) {
    initList.forEach(g => ensureCard(g.id, g.func));
    document.getElementById('glyph-count').textContent = `(${initList.length})`;
  }
  Object.entries(initGlyphs).forEach(([gid,st]) => setStatus(gid, st));
  if (ws) ws.close();
  ws = new WebSocket(`${WSB}/events/${id}`);
  ws.onmessage = e => handle(JSON.parse(e.data));
  ws.onclose = async () => {
    document.getElementById('btn-stop').disabled = true;
    // Busca status final para garantir que job-info está correto
    const r = await fetch(`${SRV}/status/${id}`).catch(()=>null);
    if (!r||!r.ok) return;
    const d = await r.json();
    Object.entries(d.glyphs||{}).forEach(([gid,st]) => setStatus(gid, st));
    document.getElementById('job-info').textContent =
      `job ${id.slice(0,8)}… | ${d.status} | ${d.device}`;
  };
}

function handle(e) {
  const ji = document.getElementById('job-info');
  switch(e.type) {
    case 'glyph_start':
      ensureCard(e.glyph_id, e.func);
      setStatus(e.glyph_id,'running');
      ji.textContent = `job ${jobId.slice(0,8)}… | ${e.func}`;
      log(`→ ${e.func} (${e.glyph_id})`, 'lg-gs'); break;
    case 'glyph_done':
      setStatus(e.glyph_id, e.status);
      if (e.status==='done') log(`✓ ${e.glyph_id} concluído`, 'lg-gd');
      else if (e.status==='error') log(`✗ ${e.glyph_id} erro`, 'lg-er');
      else if (e.status==='skipped') log(`⟳ ${e.glyph_id} pulado`, 'lg-gd'); break;
    case 'glyph_paused':
      setStatus(e.glyph_id,'paused');
      showBanner(e.glyph_id, e.func);
      log(`⏸ PAUSADO em ${e.func||e.glyph_id}`, 'lg-pa'); break;
    case 'glyph_resumed':
      setStatus(e.glyph_id,'running');
      hideBanner();
      log(`▶ retomado ${e.glyph_id}`, 'lg-gs'); break;
    case 'log':
      if (e.line && e.line.trim()) log(e.line); break;
    case 'show_image':
      if (e.glyph_id) addPreview(e.glyph_id, e.path); break;
    case 'finished':
      ji.textContent = `job ${jobId.slice(0,8)}… | ${e.returncode===0?'✓ concluído':'✗ erro'} em ${e.elapsed}s`;
      log(`━ finalizado (rc=${e.returncode}, ${e.elapsed}s)`, e.returncode===0?'lg-fi':'lg-er');
      document.getElementById('btn-stop').disabled = true;
      hideBanner(); break;
    case 'stopped':
      ji.textContent = `job ${jobId.slice(0,8)}… | parado`;
      log('■ job parado','lg-er');
      document.getElementById('btn-stop').disabled = true;
      hideBanner(); break;
    case 'error':
      log(`✗ ${e.message}`,'lg-er'); break;
  }
}

function ensureCard(gid, func) {
  if (document.getElementById(`card-${gid}`)) return;
  const c = document.getElementById('glyph-cards');
  const d = document.createElement('div');
  d.className = 'glyph-card'; d.id = `card-${gid}`; d.dataset.status = 'ready';
  d.innerHTML = `<div class="gdot"></div>
    <div class="ginfo"><div class="gfunc">${func||gid}</div><div class="gid">id: ${gid}</div></div>
    <button class="btn-bp" id="bp-${gid}" onclick="toggleBp('${gid}')" title="Breakpoint">🔴</button>
    <span id="prev-${gid}"></span>`;
  c.appendChild(d);
}

function setStatus(gid, st) {
  ensureCard(gid, gid);
  const card = document.getElementById(`card-${gid}`);
  if (card) card.dataset.status = st;
}

async function toggleBp(gid) {
  const active = breakpoints.has(gid);
  const btn = document.getElementById(`bp-${gid}`);
  // Se job ativo, sincroniza com o servidor
  if (jobId) {
    const r = await fetch(`${SRV}/breakpoint/${jobId}/${gid}`, {method: active?'DELETE':'POST'});
    if (!r.ok) return;
  }
  // Atualiza estado local sempre (funciona também antes do Run)
  active ? breakpoints.delete(gid) : breakpoints.add(gid);
  if (btn) { btn.textContent = active?'🔴':'🟡'; btn.classList.toggle('active',!active); }
}

function showBanner(gid, func) {
  document.getElementById('pause-msg').textContent = `⏸ Pausado antes de ${func||gid} (glyph ${gid})`;
  document.getElementById('pause-banner').classList.add('visible');
}
function hideBanner() { document.getElementById('pause-banner').classList.remove('visible'); }

async function continueJob() {
  if (jobId) await fetch(`${SRV}/continue/${jobId}`, {method:'POST'});
}
async function stopJob() {
  if (jobId) await fetch(`${SRV}/stop/${jobId}`, {method:'POST'});
}

// Parse .wksp client-side e mostra glyphs antes do Run
async function previewWksp() {
  const fi = document.getElementById('wksp-file');
  if (!fi.files.length) return;
  const content = await fi.files[0].text();
  const glyphs = parseWksp(content);
  document.getElementById('glyph-cards').innerHTML = '';
  document.getElementById('log').innerHTML = '';
  document.getElementById('glyph-count').textContent = '';
  // mantém breakpoints pendentes entre trocas de arquivo
  const oldBps = new Set(breakpoints);
  breakpoints.clear();
  hideBanner();
  glyphs.forEach(g => {
    ensureCard(g.id, g.func);
    if (oldBps.has(g.id)) {
      breakpoints.add(g.id);
      const btn = document.getElementById(`bp-${g.id}`);
      if (btn) { btn.textContent = '🟡'; btn.classList.add('active'); }
    }
  });
  document.getElementById('glyph-count').textContent = `(${glyphs.length})`;
}

function parseWksp(content) {
  const glyphs = [];
  for (const line of content.split('\n')) {
    const s = line.trim();
    if (!s.startsWith('Glyph:') && !s.startsWith('ProcedureBegin:')) continue;
    const parts = s.split(':');
    if (parts.length < 6) continue;
    const func = parts[2];
    const gid  = parts[3] === '' ? parts[5] : parts[4];
    glyphs.push({id: gid, func});
  }
  return glyphs;
}

async function runWorkflow() {
  const fi = document.getElementById('wksp-file');
  const device = document.getElementById('dev-sel').value;
  if (!fi.files.length) { alert('Selecione um arquivo .wksp'); return; }
  const content = await fi.files[0].text();
  const r = await fetch(`${SRV}/run`, {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({wksp_content: content, device, breakpoints: [...breakpoints]})
  });
  if (!r.ok) { const e=await r.json(); log(`✗ ${e.error||'erro'}`, 'lg-er'); return; }
  const d = await r.json();
  // keepCards=true: não apaga os cards do previewWksp nem os breakpoints visuais
  connectJob(d.job_id, {}, [], true, 'running');
}

function addPreview(gid, path) {
  const span = document.getElementById(`prev-${gid}`);
  if (!span) return;
  const src = `${SRV}/preview/${path.replace(/^\\//, '')}`;
  const img = document.createElement('img');
  img.src=src; img.className='gpreview'; img.title=path;
  img.onclick = () => window.open(src,'_blank');
  span.innerHTML = ''; span.appendChild(img);
}

function log(text, cls) {
  const el = document.getElementById('log');
  const d = document.createElement('div');
  d.className = cls||'lg'; d.textContent = text;
  el.appendChild(d); el.scrollTop = el.scrollHeight;
}

init();

// Polling para detectar novo job (1s para não perder workflows rápidos)
setInterval(async () => {
  if (jobId && ws && ws.readyState === WebSocket.OPEN) return;
  const r = await fetch(`${SRV}/current`).catch(()=>null);
  if (!r||!r.ok) return;
  const d = await r.json();
  if (d.job_id && d.job_id !== jobId)
    connectJob(d.job_id, d.glyphs||{}, d.glyph_list||[], false, d.status);
}, 1000);
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run", status_code=202)
async def run(body: RunRequest):
    global _job
    async with _job_lock:
        if _job and _job.status == "running":
            return JSONResponse(
                status_code=409,
                content={"error": "job_running", "job_id": _job.job_id},
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".wksp", delete=False, mode="w")
        tmp.write(body.wksp_content)
        tmp.close()

        glyph_list = _build_func_map(body.wksp_content)

        job_id = str(uuid.uuid4())
        _job = JobState(
            job_id=job_id,
            device=body.device,
            status="running",
            process=None,
            wksp_tmp=tmp.name,
            glyph_list=glyph_list,
            breakpoints=set(body.breakpoints),
        )
        for gid in body.breakpoints:
            open(_break_path(job_id, gid), "w").close()

    asyncio.create_task(_run_job(_job))
    return {"job_id": job_id, "status": "running"}


@app.get("/current")
async def current():
    if not _job:
        return {"job_id": None, "status": "idle"}
    return {
        "job_id": _job.job_id,
        "status": _job.status,
        "device": _job.device,
        "current_glyph": _job.current_glyph,
        "glyphs": dict(_job.glyph_status),
        "glyph_list": _job.glyph_list,
    }


@app.get("/status/{job_id}")
async def status(job_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job_id,
        "status": _job.status,
        "device": _job.device,
        "current_glyph": _job.current_glyph,
        "glyphs": dict(_job.glyph_status),
        "glyph_list": _job.glyph_list,
    }


@app.post("/stop/{job_id}")
async def stop(job_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    if _job.status not in ("running", "paused"):
        raise HTTPException(status_code=409, detail="job not running")

    if _job.process and _job.process.poll() is None:
        _job.process.terminate()

    _job.status = "stopped"
    await _emit(_job, {"type": "stopped", "reason": "user_request"})
    return {"ok": True}


@app.post("/stop/{job_id}/glyph/{glyph_id}")
async def stop_glyph(job_id: str, glyph_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    open(_skip_path(job_id, glyph_id), "w").close()
    return {"ok": True, "glyph_id": glyph_id}


@app.post("/breakpoint/{job_id}/{glyph_id}")
async def set_breakpoint(job_id: str, glyph_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    _job.breakpoints.add(glyph_id)
    open(_break_path(job_id, glyph_id), "w").close()
    return {"ok": True, "breakpoint": glyph_id}


@app.delete("/breakpoint/{job_id}/{glyph_id}")
async def remove_breakpoint(job_id: str, glyph_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    _job.breakpoints.discard(glyph_id)
    _cleanup_tmp(_break_path(job_id, glyph_id))
    return {"ok": True}


@app.post("/continue/{job_id}")
async def continue_job(job_id: str):
    if not _job or _job.job_id != job_id:
        raise HTTPException(status_code=404, detail="job not found")
    if _job.status != "paused":
        raise HTTPException(status_code=409, detail="job not paused")
    open(_continue_path(job_id), "w").close()
    return {"ok": True}


@app.get("/preview/{file_path:path}")
async def preview_image(file_path: str):
    full = os.path.realpath("/" + file_path)
    if not full.startswith("/tmp/"):
        raise HTTPException(status_code=403, detail="Acesso negado")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(full)


@app.websocket("/events/{job_id}")
async def ws_events(websocket: WebSocket, job_id: str):
    if not _job or _job.job_id != job_id:
        await websocket.close(code=4004)
        return

    await websocket.accept()

    for event in list(_job.events):
        await websocket.send_json(event)

    if _job.status not in ("running", "paused"):
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    _job.ws_queues.append(queue)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event["type"] in ("finished", "stopped", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if queue in _job.ws_queues:
            _job.ws_queues.remove(queue)
    await websocket.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
