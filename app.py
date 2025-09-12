from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from jose import jwt, JWTError

# --------- OPENAI opcional ----------
OPENAI_ENABLED = False
try:
    from openai import OpenAI  # openai>=1.40.0
    OPENAI_ENABLED = True
except Exception:
    OPENAI_ENABLED = False

# --------- PYPDF para extracción liviana ----------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore

# ===================== Configuración =====================

class Settings(BaseSettings):
    # Seguridad
    JWT_SECRET: str = os.getenv("JWT_SECRET", "cambia-esto")
    BASE_URL: str = os.getenv("BASE_URL", "")

    # Límites de subida
    UPLOAD_LIMIT: int = 20 * 1024 * 1024    # 20 MB por archivo
    TOTAL_LIMIT: int = 100 * 1024 * 1024    # 100 MB por request
    MAX_UPLOAD_FILES: int = 5

    # Almacenamiento
    STORAGE_DIR: str = "storage"
    APP_NAME: str = "Asesor Financiero"

    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    class Config:
        extra = "ignore"

S = Settings()
APP_NAME_SAFE = S.APP_NAME

# ===================== App & CORS =====================

app = FastAPI(title=S.APP_NAME)
allowed = {"http://localhost:8000", "http://127.0.0.1:8000"}
if S.BASE_URL:
    allowed.add(S.BASE_URL)
    if S.BASE_URL.startswith("https://"):
        allowed.add(S.BASE_URL.replace("https://","http://"))
    if S.BASE_URL.startswith("http://"):
        allowed.add(S.BASE_URL.replace("http://","https://"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(allowed) if allowed else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Utilidades =====================

def _make_uid(gmail: str) -> str:
    return hashlib.sha256((gmail or "").lower().encode()).hexdigest()[:16]

def _user_dir(uid: str) -> Path:
    base = Path(S.STORAGE_DIR) / uid
    (base / "docs").mkdir(parents=True, exist_ok=True)
    (base / "cache").mkdir(parents=True, exist_ok=True)
    return base

def _require_user(request: Request) -> str:
    token = None
    ah = request.headers.get("Authorization","")
    if ah.lower().startswith("bearer "):
        token = ah.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("token")
    if not token:
        raise HTTPException(401, "No autenticado")
    try:
        data = jwt.decode(token, S.JWT_SECRET, algorithms=["HS256"])
        uid = data.get("uid")
        exp = data.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > float(exp):
            raise HTTPException(401, "Token expirado")
        if not uid:
            raise HTTPException(401, "Token inválido")
        return str(uid)
    except JWTError:
        raise HTTPException(401, "Token inválido")

# --------- extracción simple de años + snippets ----------
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")  # <-- no-capturante (arregla bug) :contentReference[oaicite:6]{index=6}

def _extract_years_and_snippets(pdf_path: Path, max_per_year: int = 3) -> Dict[str, List[str]]:
    """
    Devuelve { "2020": [snippet,...], "2021":[...], ... }
    Usa pypdf; si no está disponible, retorna vacío.
    """
    result: Dict[str, List[str]] = {}
    if PdfReader is None:
        return result
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            txt = (page.extract_text() or "").strip()
            if not txt:
                continue
            # normaliza espacios
            txt = " ".join(txt.split())
            # busca líneas “contexto” por cada año encontrado
            for m in YEAR_RE.finditer(txt):
                y = m.group(0)
                # genera un snippet alrededor (0..200 chars desde el match)
                start = max(0, m.start() - 120)
                end   = min(len(txt), m.end() + 120)
                snip  = txt[start:end]
                arr = result.setdefault(y, [])
                if len(arr) < max_per_year:
                    arr.append(snip)
    except Exception:
        return result
    return result

def _save_meta(base: Path, doc: str, year_snips: Dict[str, List[str]]) -> None:
    """Guarda meta.json con estructura {year: {doc: [snips...]}} acumulativa."""
    meta_path = base / "cache" / "meta.json"
    data: Dict[str, Dict[str, List[str]]] = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for y, snips in year_snips.items():
        bucket = data.setdefault(y, {})
        bucket.setdefault(doc, [])
        # agrega sin duplicar
        for s in snips:
            if s not in bucket[doc]:
                bucket[doc].append(s)
    meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def _read_meta(base: Path) -> Dict[str, Dict[str, List[str]]]:
    meta_path = base / "cache" / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _parse_years(text: str) -> List[int]:
    """Detecta años individuales o implícitamente un rango si hay ≥2 (min..max)."""
    nums = [int(x) for x in YEAR_RE.findall(text)]  # no-capturante, año completo :contentReference[oaicite:7]{index=7}
    if not nums:
        return []
    if len(nums) >= 2:
        a, b = min(nums), max(nums)
        return list(range(a, b + 1))
    return nums

# ===================== HTML del portal (UI modernizada) =====================

PORTAL_HTML = f"""
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{APP_NAME_SAFE}</title>
  <link rel="preconnect" href="https://cdn.jsdelivr.net" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{ --bg:#0b1220; --card:#121a2b; --muted:#9fb1d3; --acc:#4f8cff; --ok:#e6ffed; --warn:#fff8db; --bad:#ffecec; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; background:var(--bg); color:#e8eefc; }}
    header {{ display:flex; align-items:center; justify-content:space-between; padding:16px 20px; position:sticky; top:0; background:rgba(11,18,32,.8); backdrop-filter: blur(8px); border-bottom:1px solid #1d2840; }}
    h1 {{ font-size:16px; margin:0; letter-spacing:.4px; }}
    main {{ padding:20px; display:grid; gap:20px; grid-template-columns: 1fr; max-width:1200px; margin:0 auto; }}
    .card {{ background:var(--card); border:1px solid #1d2840; border-radius:14px; padding:16px; }}
    .row {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); }}
    .btn {{ background:var(--acc); color:white; border:none; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:600; }}
    .btn:disabled {{ opacity:.6; cursor:not-allowed; }}
    .muted {{ color:var(--muted); font-size:12px; }}
    .nowrap {{ white-space:nowrap; display:inline-block; }} /* Prosa premium en 1 línea */  /* :contentReference[oaicite:8]{index=8} */
    ul#pending {{ list-style:none; padding:0; margin:8px 0 0 0; }}
    #pending li {{ font-size:13px; margin:2px 0; }}
    #pending a {{ color:#ff7b7b; text-decoration:none; margin-left:8px; }}
    #charts {{ display:grid; gap:16px; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); }}
    #kpis {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); }}
    .kpi {{ padding:12px; border-radius:12px; border:1px solid #203052; background:#0e1626; }}
    .kpi.ok {{ background: var(--ok); color:#0a3a12; }}
    .kpi.warn {{ background: var(--warn); color:#5a4a00; }}
    .kpi.bad {{ background: var(--bad); color:#5a0000; }}
    textarea {{ width:100%; min-height:120px; border-radius:8px; border:1px solid #203052; background:#0e1626; color:#e8eefc; padding:10px; }}
    @media print {{ header,#controls,#uploadBtn,#askBtn,#autoBtn,#premium {{ display:none !important; }} }}
  </style>
</head>
<body>
<header>
  <h1>{APP_NAME_SAFE}</h1>
  <button id="printBtn" class="btn">🖨️ Imprimir</button>
</header>
<main>
  <section class="card" id="controls">
    <div class="row">
      <div>
        <div><label for="filepick" class="btn">Elegir PDF(s)</label> <span class="muted">máx {S.MAX_UPLOAD_FILES} por subida</span></div>
        <input id="filepick" type="file" accept="application/pdf" multiple style="display:none" />
        <ul id="pending"></ul>
        <button id="uploadBtn" class="btn">Subir PDFs e indexar</button>
        <div id="uploadMsg" class="muted"></div>
      </div>
      <div>
        <label class="nowrap" id="premium"><input type="checkbox" id="prosaChk" /> Prosa premium (IA)</label>
        <textarea id="questions" placeholder="Escribe hasta 5 preguntas, una por línea"></textarea>
        <button id="askBtn" class="btn">Preguntar</button>
      </div>
      <div>
        <button id="autoBtn" class="btn">Generar Reporte Automático</button>
      </div>
    </div>
  </section>

  <section class="card" id="answers"></section>
  <section class="card" id="report">
    <div id="charts"></div>
    <div id="kpis"></div>
    <div id="gloss"></div>
  </section>
</main>

<script>
const $ = s => document.querySelector(s);
const pending = [];
$("#filepick").onchange = () => {
  for (const f of $("#filepick").files) {
    if (!pending.find(p => p.name === f.name)) pending.push(f);
  }
  renderPending(); $("#filepick").value='';
};
function renderPending(){
  $("#pending").innerHTML = pending.map(p => `<li>${p.name} <a href="#" data-n="${p.name}">✖</a></li>`).join('');
  $("#pending").querySelectorAll('a').forEach(a=>{
    a.onclick = (e)=>{ e.preventDefault(); const n=a.dataset.n; const i=pending.findIndex(p=>p.name===n); if(i>=0) pending.splice(i,1); renderPending(); };
  });
}
function authHeader(){ const t = localStorage.getItem('token'); return t ? { 'Authorization': 'Bearer '+t } : {}; }

$("#uploadBtn").onclick = async()=>{
  if(!pending.length){ alert('No hay archivos para subir'); return; }
  const fd = new FormData(); pending.forEach(f=>fd.append('files',f));
  $("#uploadBtn").disabled = true;
  try{
    const r = await fetch('/upload', { method:'POST', headers: authHeader(), body: fd, credentials:'include' });
    const data = await r.json();
    if(data.saved){ for(const nm of data.saved){ const i=pending.findIndex(p=>p.name===nm); if(i>=0) pending.splice(i,1); } renderPending(); }
    $("#uploadMsg").textContent = data.message || (data.ok ? 'Subida realizada' : 'Error al subir');
  }catch{ $("#uploadMsg").textContent='Error de red al subir.'; }
  $("#uploadBtn").disabled = false;
};

$("#askBtn").onclick = async()=>{
  const qs = $("#questions").value.split(/\\n+/).map(x=>x.trim()).filter(Boolean).slice(0,5);
  if(!qs.length){ alert('Ingresa al menos una pregunta'); return; }
  const prosa = $("#prosaChk").checked;
  $("#answers").innerHTML = 'Consultando...';
  try{
    const r = await fetch('/ask', { method:'POST', headers: { 'Content-Type':'application/json', ...authHeader() }, body: JSON.stringify({ questions: qs, prosa }) });
    const data = await r.json();
    if(!Array.isArray(data)){ $("#answers").textContent = data.message || 'Error al preguntar.'; return; }
    $("#answers").innerHTML = data.map(x => `<div style="margin-bottom:12px"><b>${x.question}</b><br>${x.answer}</div>`).join('');
  }catch{ $("#answers").textContent='Error de red.'; }
};

$("#autoBtn").onclick = async()=>{
  const prosa = $("#prosaChk").checked ? 1 : 0;
  $("#report").querySelector('#charts').innerHTML=''; $("#report").querySelector('#kpis').innerHTML=''; $("#report").querySelector('#gloss').innerHTML='';
  try{
    const r = await fetch('/auto-report?prosa='+prosa, { headers: authHeader(), credentials:'include' });
    const data = await r.json();
    if(data.message){ $("#report").insertAdjacentHTML('afterbegin','<p>'+data.message+'</p>'); }
    const charts = data.charts || [];
    const chartsDiv = $("#charts");
    charts.forEach((cfg,i)=>{ const c=document.createElement('canvas'); chartsDiv.appendChild(c); new Chart(c.getContext('2d'), cfg.config); });
    const kpisDiv = $("#kpis");
    (data.kpis||[]).forEach(k=>{ const d=document.createElement('div'); d.className='kpi '+(k.state||''); d.innerHTML = `<b>${k.name_es} / ${k.name_en}</b><br>Valor: ${k.value}${k.unit||''}<br>Fórmula: ${k.formula}<br>Evaluación: ${k.state}<br>Acción sugerida: ${k.action}`; kpisDiv.appendChild(d); });
    const glossDiv = $("#gloss"); const g = data.glossary||{}; if(Object.keys(g).length){ let html='<h3>Glosario</h3><ul>'; Object.keys(g).forEach(k=> html+=`<li><b>${k}</b>: ${g[k]}</li>` ); html+='</ul>'; glossDiv.innerHTML=html; }
  }catch{ $("#report").insertAdjacentHTML('afterbegin','<p>Error de red.</p>'); }
};

$("#printBtn").onclick = ()=> window.print();
</script>
</body>
</html>
"""

# ===================== Rutas =====================

@app.get("/portal", response_class=HTMLResponse)
def portal_page(request: Request):
    _require_user(request)
    return HTMLResponse(PORTAL_HTML)  # UI con “Prosa premium (IA)” en una sola línea :contentReference[oaicite:9]{index=9}

@app.get("/__warmup")
def warmup():
    return {"ok": True}

# ---------- Upload ----------

def _index_worker(base: Path, saved: List[str]) -> None:
    try:
        for nm in saved:
            year_snips = _extract_years_and_snippets(base / "docs" / nm)
            if year_snips:
                _save_meta(base, nm, year_snips)
    except Exception as e:
        print("index_worker error:", e, flush=True)

@app.post("/upload")
async def upload(request: Request, background_tasks: BackgroundTasks, files: Optional[List[UploadFile]] = File(None)):
    uid = _require_user(request)
    base = _user_dir(uid)

    if not files:
        return {"ok": False, "message": "Selecciona al menos un PDF."}
    if len(files) > S.MAX_UPLOAD_FILES:
        return {"ok": False, "message": f"Máximo {S.MAX_UPLOAD_FILES} PDFs por subida."}

    total = 0
    saved: List[str] = []
    for f in files:
        b = await f.read()
        total += len(b)
        if len(b) > S.UPLOAD_LIMIT:
            return {"ok": False, "message": f"{f.filename}: supera {S.UPLOAD_LIMIT//(1024*1024)} MB"}
        if total > S.TOTAL_LIMIT:
            return {"ok": False, "message": f"Superaste {S.TOTAL_LIMIT//(1024*1024)} MB totales por subida"}
        name = (f.filename or "archivo.pdf").strip()
        if not name.lower().endswith(".pdf"):
            return {"ok": False, "message": f"{name}: solo se aceptan PDF"}
        (base / "docs" / name).write_bytes(b)
        saved.append(name)

    background_tasks.add_task(_index_worker, base, saved)
    msg = f"Estamos cargando tus archivo(s) PDF: {', '.join('\"'+n+'\"' for n in saved)}. Analizándolos en este instante."
    return {"ok": True, "saved": saved, "message": msg}  # mensaje amable :contentReference[oaicite:10]{index=10}

# ---------- Ask (con OpenAI opcional) ----------

def _build_context_from_meta(base: Path, years: List[int]) -> str:
    meta = _read_meta(base)  # {year: {doc: [snips...]}}
    lines: List[str] = []
    if years:
        for y in years:
            bucket = meta.get(str(y), {})
            if not bucket:
                lines.append(f"[{y}] Sin evidencia en PDFs.")
                continue
            docs = ", ".join(sorted(bucket.keys()))
            lines.append(f"[{y}] Documentos: {docs}")
            # toma hasta 2 snippets por doc
            for d, snips in list(bucket.items())[:5]:
                for s in snips[:2]:
                    lines.append(f"({y}) {d}: {s}")
    else:
        # general: lista de docs y algunos snippets variados
        for y, bucket in list(meta.items())[:6]:
            docs = ", ".join(sorted(bucket.keys()))
            lines.append(f"[{y}] Documentos: {docs}")
            for d, snips in list(bucket.items())[:1]:
                for s in snips[:1]:
                    lines.append(f"({y}) {d}: {s}")
    return "\n".join(lines) if lines else "No hay evidencia en PDFs."

@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    questions: List[str] = (body.get("questions") or [])[:5]
    prosa: bool = bool(body.get("prosa"))
    uid = _require_user(request)
    base = _user_dir(uid)

    if not questions:
        return []

    answers: List[Dict[str, str]] = []
    for q in questions:
        yrs = _parse_years(q)
        # construir respuesta básica por evidencia
        ctx = _build_context_from_meta(base, yrs)

        if prosa and OPENAI_ENABLED and S.OPENAI_API_KEY:
            try:
                client = OpenAI(api_key=S.OPENAI_API_KEY)
                sys = (
                    "Eres un analista financiero. Responde en español, claro y directo. "
                    "Si el usuario menciona años, estructura la respuesta por cada año explícitamente. "
                    "Usa solo la evidencia provista; si falta, dilo."
                )
                user = f"Pregunta: {q}\n\nEvidencia de PDFs:\n{ctx}"
                resp = client.chat.completions.create(
                    model=S.OPENAI_MODEL,
                    messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                    temperature=0.1,
                )
                out = resp.choices[0].message.content.strip() if resp and resp.choices else "No hay respuesta."
                answers.append({"question": q, "answer": out})
                continue
            except Exception as e:
                # fallback a respuesta básica
                pass

        # respuesta básica (sin OpenAI)
        if yrs:
            # listar por año
            meta = _read_meta(base)
            parts = []
            for y in yrs:
                bucket = meta.get(str(y), {})
                if bucket:
                    docs = ", ".join(sorted(bucket.keys()))
                    snippet = "; ".join((bucket[list(bucket.keys())[0]] or [""])[:1])
                    parts.append(f"Para el año {y}: Documentos {docs}. Ejemplo: {snippet[:260]}...")
                else:
                    parts.append(f"Para el año {y}: no se encontró evidencia en tus PDFs.")
            answers.append({"question": q, "answer": "<br>".join(parts)})
        else:
            answers.append({"question": q, "answer": "No se especificaron años. Puedes indicar un año o rango (p. ej., 2020–2023)."})
    return answers

# ---------- Auto-report (con narrativa IA opcional) ----------

@app.get("/auto-report")
async def auto_report(request: Request, prosa: int = 0):
    uid = _require_user(request)
    base = _user_dir(uid)

    # Datos demo (puedes luego calcularlos desde PDFs)
    years   = [2020, 2021, 2022, 2023]
    revenue = [120, 135, 150, 180]
    ebitda  = [18, 22, 25, 28]
    margin  = [round(e/r*100,1) for e,r in zip(ebitda,revenue)]
    cost_mix = [40, 50, 10]  # Fijos, Variables, Otros

    # 1) Barras: Ingresos + EBITDA
    charts: List[Dict[str, Any]] = [{
        "config": {
            "type": "bar",
            "data": {
                "labels": years,
                "datasets": [
                    {"label":"Ingresos / Revenue","data":revenue},
                    {"label":"EBITDA / EBITDA","data":ebitda},
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Ingresos y EBITDA"}},
                "scales": {"x":{"title":{"display":True,"text":"Año / Year"}},
                           "y":{"title":{"display":True,"text":"Valor"}}}
            }
        }
    }]

    # 2) Línea: Margen EBITDA
    charts.append({
        "config": {
            "type": "line",
            "data": {"labels": years, "datasets": [{"label":"Margen EBITDA (%) / EBITDA Margin (%)","data":margin}]},
            "options": {"responsive":True,"plugins":{"title":{"display":True,"text":"Margen EBITDA"}},
                        "scales":{"x":{"title":{"display":True,"text":"Año / Year"}},"y":{"title":{"display":True,"text":"%"}}}}
        }
    })

    # 3) Pie: Mezcla de costos
    charts.append({
        "config": {
            "type":"pie",
            "data":{"labels":["Costos Fijos / Fixed Costs","Costos Variables / Variable Costs","Otros / Other"],
                    "datasets":[{"data":cost_mix}]},
            "options":{"responsive":True,"plugins":{"title":{"display":True,"text":"Composición de Costos / Cost Composition"}}}
        }
    })

    # 4) Línea: Crecimiento Ingresos
    growth = [0] + [round((revenue[i]-revenue[i-1])/revenue[i-1]*100,2) for i in range(1,len(revenue))]
    charts.append({
        "config": {
            "type":"line",
            "data":{"labels":years,"datasets":[{"label":"Crecimiento de Ingresos (%) / Revenue Growth (%)","data":growth}]},
            "options":{"responsive":True,"plugins":{"title":{"display":True,"text":"Crecimiento de Ingresos"}},
                       "scales":{"x":{"title":{"display":True,"text":"Año / Year"}},"y":{"title":{"display":True,"text":"%"}}}}
        }
    })

    # KPI con semáforo
    def state(val, good, warn):
        return "ok" if val>=good else ("warn" if val>=warn else "bad")

    kpis = [
        {"name_es":"Margen EBITDA","name_en":"EBITDA Margin","value":margin[-1],"unit":"%","formula":"EBITDA / Ingresos",
         "state":state(margin[-1],20,12),"action":"Revisar pricing y disciplina de costos."},
        {"name_es":"Crec. Ingresos","name_en":"Revenue Growth","value":growth[-1],"unit":"%","formula":"(Ingresos_t-Ingresos_t-1)/Ingresos_t-1",
         "state":state(growth[-1],10,0),"action":"Expandir canales y mix de productos."},
        {"name_es":"Costos Fijos","name_en":"Fixed Cost Ratio","value":round(cost_mix[0]/sum(cost_mix)*100,2),"unit":"%","formula":"Fijos / Totales",
         "state":state(round(cost_mix[0]/sum(cost_mix)*100,2)<=50 and 100 or 0,50,70),"action":"Optimizar estructura de costos fijos."},
    ]

    glossary = {
        "EBITDA":"Ganancias antes de Intereses, Impuestos, Depreciación y Amortización / Earnings Before Interest, Taxes, Depreciation and Amortization",
        "ROI":"Retorno sobre la Inversión / Return on Investment",
        "KPI":"Indicador Clave de Desempeño / Key Performance Indicator",
        "WACC":"Costo Promedio Ponderado de Capital / Weighted Average Cost of Capital",
    }

    payload = {"charts": charts, "kpis": kpis, "glossary": glossary}

    # Narrativa premium con OpenAI (opcional)
    if prosa and OPENAI_ENABLED and S.OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=S.OPENAI_API_KEY)
            sys = ("Eres un analista financiero. Genera un breve comentario en español, claro y accionable, "
                   "sobre la evolución de Ingresos, EBITDA, Margen, Crecimiento y mezcla de costos. "
                   "Incluye recomendaciones concretas.")
            user = f"Series: years={years}, revenue={revenue}, ebitda={ebitda}, margin={margin}, growth={growth}, cost_mix={cost_mix}"
            resp = client.chat.completions.create(
                model=S.OPENAI_MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0.2,
            )
            note = resp.choices[0].message.content.strip() if resp and resp.choices else ""
            if note:
                payload["message"] = note
        except Exception:
            # si falla OpenAI, seguimos sin narrativa
            pass

    return JSONResponse(payload)
