#!/usr/bin/env python3
"""
Agente Financiero en la Nube – versión Mercado Pago (Checkout Pro)

✔ Pago con Mercado Pago (pase de 1 día)
✔ Captura de Nombre, Apellido y Gmail (valida que sea @gmail.com)
✔ Verificación de pago al regresar de Checkout (Payments API)
✔ Subida de PDFs, RAG por usuario y Reporte KPI (igual que antes)

Cómo usar:
1) Reemplaza tu app.py por este archivo (o guárdalo como app.py).
2) En requirements.txt agrega: mercadopago
3) Variables de entorno nuevas (Render → Environment):
   MP_ACCESS_TOKEN=APP_USR-xxxxxxxxxxxxxxxxxxxxxxx   # token privado
   MP_PUBLIC_KEY=TEST-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # clave pública (por si luego usas Bricks)
   MP_PRICE_1DAY=10000                               # precio numérico (ej. CLP)
   MP_CURRENCY=CLP                                   # moneda (CLP, USD, etc.)
   BASE_URL=https://TU-URL.onrender.com              # tu URL pública
   APP_NAME=Asesor Financiero 1‑día
   JWT_SECRET=un_super_secreto

Notas:
- Creamos una Preferencia de pago (Checkout Pro) y redirigimos a su init_point.
- Configuramos back_urls + auto_return=approved y verificamos el pago con Payments API
  usando payment_id. (Ver docs oficiales de Preferencias y back_urls).
"""
from __future__ import annotations
import os, re, json, hashlib, shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
from typing import Optional

import mercadopago
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from jose import jwt, JWTError

# ===================== Config =====================
class Settings(BaseSettings):
    MP_ACCESS_TOKEN: str
    MP_PUBLIC_KEY: str
    MP_PRICE_1DAY: float
    MP_CURRENCY: str = "CLP"

    JWT_SECRET: str
    BASE_URL: str = "http://localhost:8000"
    APP_NAME: str = "Asesor Financiero 1‑día"

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    STORAGE_DIR: str = "storage"
    MODEL_NAME: str = "intfloat/multilingual-e5-small"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 200
    MIN_CHARS: int = 200
    TOP_K: int = 6
    NORMALIZE: bool = True
    MAX_UPLOAD_FILES: int = 5
    SINGLE_FILE_MAX_MB: int = 20
    MAX_TOTAL_MB: int = 100

    class Config:
        env_file = ".env"

# Cargar settings con manejo de error para que /health lo muestre si falla
SETTINGS_ERROR = None
try:
    settings = Settings()
except Exception as e:
    SETTINGS_ERROR = str(e)
    settings = None

APP_NAME_SAFE = (settings.APP_NAME if settings else "Asesor Financiero")

app = FastAPI(title=(settings.APP_NAME if settings else "Finance"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Warm-up: precargar el modelo en segundo plano al iniciar ----
from threading import Thread

@app.on_event("startup")
def _warm_start():
    def _preload():
        try:
            print(">> Precargando modelo de embeddings...")
            get_model()  # descarga/carga el modelo una vez
            print(">> Modelo listo.")
        except Exception as e:
            print("!! Error precargando modelo:", e)
    Thread(target=_preload, daemon=True).start()

# Endpoint manual por si quieres forzarlo desde el navegador: /__warmup
@app.get("/__warmup")
def __warmup():
    get_model()
    return {"ok": True, "msg": "modelo listo"}


# Mercado Pago SDK
mp = mercadopago.SDK(settings.MP_ACCESS_TOKEN) if settings else None
# --- Cupones propios del comercio ---
COUPONS = {
    "INVESTU-100": {"type": "free", "desc": "Acceso gratis"},
    "INVESTU-50":  {"type": "percent", "value": 50, "desc": "50% OFF"},
}


# ===================== Embeddings & RAG (igual) =====================
_model = None
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # import aquí
        _model = SentenceTransformer(settings.MODEL_NAME)
    return _model

def embed_texts(texts: List[str]):
    import numpy as np  # import aquí
    vecs = get_model().encode(texts, batch_size=32, convert_to_numpy=True,
                              normalize_embeddings=settings.NORMALIZE)
    return np.ascontiguousarray(vecs.astype("float32"))

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = "\n".join([ln.strip() for ln in t.splitlines()])
    t = " ".join(t.split())
    return t.strip()

def chunk_text(text: str) -> List[str]:
    text = text.strip()
    if not text: return []
    size, overlap = settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    chunks, start, N = [], 0, len(text)
    while start < N:
        end = min(start + size, N)
        chunk = text[start:end]
        if end < N:
            last_period = chunk.rfind('.')
            if last_period != -1 and last_period > size * 0.5:
                chunk = chunk[: last_period + 1]
                end = start + len(chunk)
        if len(chunk) >= settings.MIN_CHARS:
            chunks.append(chunk)
        start = max(end - overlap, end)
        if start == end: break
    return chunks

# ===== Paths por usuario =====

def make_user_id(gmail: str) -> str:
    return hashlib.sha256(gmail.lower().encode()).hexdigest()[:16]

def user_dir(uid: str) -> Path:
    return Path(settings.STORAGE_DIR) / uid

def ensure_dirs(base: Path):
    (base/"docs").mkdir(parents=True, exist_ok=True)
    (base/".rag_index").mkdir(parents=True, exist_ok=True)

# ===== FAISS por usuario =====

def idx_paths(base: Path) -> Dict[str, Path]:
    return {"faiss": base/".rag_index"/"index.faiss", "meta": base/".rag_index"/"metadata.jsonl"}

def load_index(base: Path):
    import faiss  # import aquí
    p = idx_paths(base)
    dim = get_model().get_sentence_embedding_dimension()
    if p["faiss"].exists() and p["meta"].exists():
        idx = faiss.read_index(str(p["faiss"]))
        meta = [json.loads(x) for x in open(p["meta"], "r", encoding="utf-8") if x.strip()]
        return idx, meta
    return faiss.IndexFlatIP(dim), []

def save_index(base: Path, idx, meta: List[Dict[str, Any]]):
    import faiss  # import aquí
    p = idx_paths(base)
    p["faiss"].parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(p["faiss"]))
    with open(p["meta"], "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def add_pdfs_to_index(base: Path, pdfs: List[Path]) -> int:
    import fitz  # PyMuPDF (import aquí)
    ensure_dirs(base)
    idx, meta = load_index(base)
    new_txt, new_meta = [], []
    for pdf in pdfs:
        if not pdf.exists(): 
            continue
        with fitz.open(pdf) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = clean_text(page.get_text("text") or "")
                if len(text) < 50:
                    continue
                for ck in chunk_text(text):
                    new_txt.append(ck)
                    new_meta.append({"doc_title": pdf.name, "page": page_num, "text": ck})
    if not new_txt:
        return 0
    vecs = embed_texts(new_txt)
    idx.add(vecs)
    meta.extend(new_meta)
    save_index(base, idx, meta)
    return len(new_txt)

def semantic_search(base: Path, q: str, k: int) -> List[Tuple[float, Dict[str, Any]]]:
    idx, meta = load_index(base)
    if idx.ntotal == 0: return []
    D, I = idx.search(embed_texts([q]), k)
    out = []
    for s, i in zip(D[0], I[0]):
        if i == -1: continue
        out.append((float(s), meta[i]))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

# ===================== Auth (pase 24h) =====================

def make_token(uid: str, hours: int = 24) -> str:
    payload = {
        "sub": uid,
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=hours)).timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

def read_token(token: str) -> str:
    try:
        data = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return data["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")


# =============== PROSA PREMIUM (opcional, requiere OPENAI_API_KEY) ===============
def _premium_on() -> bool:
    try:
        return bool(getattr(settings, "OPENAI_API_KEY", None))
    except Exception:
        return False

def _gpt(system_msg: str, user_msg: str, max_tokens: int = 700) -> str | None:
    """Llama a GPT si hay API key. Devuelve texto o None si algo falla."""
    if not _premium_on():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)  # usa timeout por defecto
        resp = client.chat.completions.create(
            model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

def premium_answer_for_question(question: str, ctx: list[dict]) -> str:
    """
    Redacción clara y accionable a partir de los fragmentos (ctx).
    - Usa Markdown.
    - Si necesitas fórmulas, usa LaTeX con $...$ o $$...$$.
    - Si hay 2+ métricas comparables, añade AL FINAL un bloque de código `chart`
      con JSON válido para Chart.js.
    """
    # evidencia (top-5)
    snips = []
    for c in (ctx or [])[:5]:
        doc = c.get("doc_title","?"); pg = c.get("page","?"); tx = c.get("text","")
        snips.append(f"[{doc} p.{pg}] {tx}")
    evidence = "\n\n".join(snips) or "(sin evidencia)"

    sys = (
        "Eres un analista financiero senior. Responde en español, claro y conciso, usando Markdown. "
        "Cuando sea útil, escribe fórmulas con LaTeX (delimitadores $...$ o $$...$$). "
        "Si detectas 2 o más métricas numéricas comparables, agrega al FINAL exactamente "
        "UN bloque de código con encabezado 'chart' que contenga JSON válido para Chart.js. "
        "No inventes datos fuera de la evidencia; si faltan, dilo."
    )
    usr = (
        f"Pregunta:\n{question}\n\n"
        f"Evidencia (usa SOLO estos datos):\n{evidence}\n\n"
        "Devuelve, en este orden:\n"
        "1) Resumen ejecutivo (3–4 líneas).\n"
        "2) Hallazgos clave (viñetas, con cifras si aparecen).\n"
        "3) Recomendaciones accionables (viñetas) y riesgos/alertas.\n"
        "4) (Opcional) Un bloque `chart` con JSON válido si hay suficientes números para comparar.\n"
    )
    out = _gpt(sys, usr, max_tokens=900)
    return out or "Pasajes más relevantes (ver 'evidence')."


def premium_exec_summary(period: str, kpis: dict) -> str | None:
    """
    Resumen ejecutivo del reporte a partir de KPIs (si hay OPENAI_API_KEY).
    - Usa Markdown y LaTeX si corresponde.
    - Incluye al FINAL un bloque `chart` (Chart.js) con 1 gráfico sencillo
      si hay datos (p. ej., ratios clave).
    """
    if not _premium_on():
        return None
    import json
    sys = (
        "Eres un consultor financiero. Redacta un resumen ejecutivo en español, "
        "tono profesional, claro, 8–12 viñetas máximo. Usa Markdown. "
        "Para fórmulas usa LaTeX con $...$ o $$...$$ cuando ayude. "
        "Si hay ratios clave (liquidez, endeudamiento, rentabilidad), añade al FINAL "
        "exactamente UN bloque `chart` con JSON válido para Chart.js comparando esos ratios."
    )
    usr = (
        f"Período: {period or 'no especificado'}\n\n"
        f"KPIs (JSON):\n{json.dumps(kpis, ensure_ascii=False)}\n\n"
        "Incluye: liquidez, endeudamiento, rentabilidad, actividad, alertas y 3–5 acciones."
    )
    return _gpt(sys, usr, max_tokens=700)
 

# ===================== HTML =====================
BASE_HTML = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>[APPNAME]</title>
<style>
  body{font-family:system-ui;margin:2rem}
  .card{max-width:860px;margin:auto;padding:1.2rem 1.5rem;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.06)}
  input,button{padding:.6rem .8rem;border-radius:10px;border:1px solid #d1d5db;width:100%}
  button{background:#111;color:#fff;border:none;cursor:pointer}
  .row{display:flex;gap:12px;flex-wrap:wrap}
  .muted{color:#6b7280}
</style>
<script>
async function startCheckout(ev){
  ev.preventDefault();
  const f  = ev.target.closest('form');
  const fd = new FormData(f);

  const nombre   = (fd.get('nombre')  || '').toString().trim();
  const apellido = (fd.get('apellido')|| '').toString().trim();
  const gmail    = (fd.get('gmail')   || '').toString().trim();
  const coupon   = (fd.get('coupon')  || '').toString().trim();

  if(!/^[^\\s@]+@gmail\\.com$/.test(gmail)){ alert('Ingresa un Gmail válido'); return; }

  const r = await fetch('/mp/create-preference', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ nombre, apellido, gmail, coupon })
  });
  const data = await r.json();

  if (data.skip === true && data.token){
    localStorage.setItem('token', data.token); // cupón 100% → entra directo
    location.href = '/portal';
    return;
  }
  if(!data.init_point){ alert('No se pudo crear la preferencia'); return; }
  location.href = data.init_point; // ir a Mercado Pago
}
</script>
</head>
<body>
<div class="card">
  <h1>[APPNAME]</h1>
  <p class="muted">Acceso por 24h tras el pago con Mercado Pago. Sube tus informes PDF y obtén KPI + análisis + sugerencias. Para asesoría completa: <b>dreamingup7@gmail.com</b>.</p>

  <form onsubmit="startCheckout(event)">
    <div class="row">
      <input name="nombre"   placeholder="Nombre" required>
      <input name="apellido" placeholder="Apellido" required>
      <input name="gmail"    placeholder="Gmail (obligatorio)" required>
      <input name="coupon"   placeholder="Cupón (opcional)">
    </div>
    <div style="margin-top:12px">
      <button>Pagar y acceder por 1 día</button>
    </div>
  </form>
</div>
</body>
</html>"""


PORTAL_HTML = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portal</title>
<style>
  :root{--muted:#6b7280}
  body{font-family:system-ui;margin:2rem;background:#fafafa}
  .card{max-width:1000px;margin:auto;padding:1.2rem 1.5rem;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.06);background:#fff}
  input,button,textarea{padding:.6rem .8rem;border-radius:10px;border:1px solid #d1d5db;width:100%;background:#fff}
  button{background:#111;color:#fff;border:none;cursor:pointer}
  pre{white-space:pre-wrap;background:#f9fafb;padding:1rem;border-radius:10px;border:1px solid #eee}
  .row{display:flex;gap:12px;flex-wrap:wrap}
  .muted{color:var(--muted)}
  .md h2,.md h3{margin:.8rem 0 .4rem}
  .pill{display:inline-block;font-size:.75rem;color:#111;background:#e5e7eb;border-radius:999px;padding:.2rem .6rem;margin-left:.4rem}
  .src{font-size:.85rem;color:var(--muted)}
  .charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:10px}
  .kpi{display:grid;grid-template-columns:1fr auto;gap:6px;padding:.6rem .8rem;border:1px solid #eee;border-radius:10px;background:#fbfbfb}
</style>

<!-- Librerías para render bonito -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.9/dist/purify.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="card">
  <h2>Portal de usuario (pase activo)</h2>

  <!-- SUBIR PDFs -->
  <form id="up" method="post" enctype="multipart/form-data">
    <input type="file" name="files" multiple accept="application/pdf" required>
    <small class="muted">Límites: máx <b>[MAX_FILES]</b> PDFs por subida · <b>[SINGLE_MAX] MB</b> cada uno · hasta <b>[TOTAL_MAX] MB</b> en total.</small>
    <div style="margin-top:8px"><button>Subir PDFs e indexar</button></div>
  </form>
  <pre id="upres"></pre>

  <!-- PREGUNTAS (hasta 5 a la vez) -->
  <form id="askf">
    <textarea name="questions" rows="4" placeholder="Escribe hasta 5 preguntas, una por línea"></textarea>
    <label style="display:inline-flex;gap:6px;align-items:center;margin-top:8px">
      <input type="checkbox" id="prosa"> Prosa premium (IA)
    </label>
    <div style="margin-top:8px"><button type="submit">Preguntar</button></div>
  </form>
  <div class="muted" style="margin:.4rem 0">Puedes hacer hasta 5 preguntas a la vez (una por línea).</div>
  <!-- 👉 contenedor “bonito” para respuestas -->
  <div id="askres" class="md"></div>

  <hr style="margin:18px 0">

  <!-- REPORTE -->
  <form id="rep">
    <input name="periodo" placeholder="Período (opcional)">
    <small class="muted">Indica meses o años a evaluar. Ej: “2022–2024”, “ene–jun 2024”, “últimos 12 meses”.</small>
    <div style="margin-top:8px"><button>Generar Reporte Automático</button></div>
  </form>

  <!-- 👉 contenedor “bonito” para reporte -->
  <div id="repres" class="md" style="margin-top:10px"></div>
  <div id="repcharts" class="charts"></div>
</div>

<script>
const MAX_FILES = [MAX_FILES];
const SINGLE_MAX = [SINGLE_MAX]; // MB por archivo
const TOTAL_MAX  = [TOTAL_MAX];  // MB por subida

function toMB(n){ return (n/1024/1024).toFixed(1) + " MB"; }

async function withToken(url,opts={}) {
  const t = localStorage.getItem('token') || '';
  opts.headers = Object.assign({'Authorization':'Bearer '+t}, opts.headers||{}, opts.headers);
  return fetch(url,opts);
}

// ------ Render Markdown + Matemáticas + Gráficos ------
function renderMD(el, md){
  try {
    const raw = marked.parse(md || "");
    const clean = DOMPurify.sanitize(raw);
    el.innerHTML = clean;
    // KaTeX
    try {
      renderMathInElement(el, {
        delimiters: [
          {left: "$$", right: "$$", display: true},
          {left: "$",  right: "$",  display: false}
        ]
      });
    } catch(e){}
    // Bloques ```chart ...``` → Chart.js
    const blocks = el.querySelectorAll("pre code.language-chart");
    blocks.forEach((code, i) => {
      const json = code.textContent.trim();
      let cfg = null;
      try { cfg = JSON.parse(json); } catch(e){ return; }
      const pre = code.closest("pre");
      const canvas = document.createElement("canvas");
      pre.replaceWith(canvas);
      new Chart(canvas.getContext('2d'), cfg);
    });
  } catch(err) {
    el.textContent = "Error renderizando Markdown: " + err;
  }
}

// ---------------- SUBIR PDFs ----------------
up.onsubmit = async e => {
  e.preventDefault();
  const input = up.querySelector('input[type=file]');
  const files = Array.from(input.files || []);
  const box = document.getElementById('upres');

  if (!files.length) { box.textContent = "Selecciona al menos un PDF."; return; }
  if (files.length > MAX_FILES) { box.textContent = `Máximo ${MAX_FILES} PDFs por subida.`; return; }

  let total = 0;
  for (const f of files) {
    total += f.size;
    if (!/\\.pdf$/i.test(f.name)) { box.textContent = `${f.name}: solo PDF`; return; }
    if (f.size > SINGLE_MAX*1024*1024) { box.textContent = `${f.name} supera ${SINGLE_MAX} MB (${toMB(f.size)})`; return; }
  }
  if (total > TOTAL_MAX*1024*1024) {
    box.textContent = `Superaste el total permitido (${TOTAL_MAX} MB). Subiste ${toMB(total)}.`;
    return;
  }

  const fd = new FormData(up);
  const r  = await withToken('/upload',{method:'POST',body:fd});
  box.textContent = `HTTP ${r.status}\\n` + await r.text();
};

// ---------------- PREGUNTAR ----------------
askf.onsubmit = async (e) => {
  e.preventDefault();
  const raw = new FormData(askf).get('questions') || '';
  const list = raw.split(/\\r?\\n/).map(s => s.trim()).filter(Boolean).slice(0, 5);
  const prosa = document.getElementById('prosa')?.checked || false;

  const box = document.getElementById('askres');
  if (!list.length) { box.textContent = 'Escribe al menos 1 pregunta (una por línea).'; return; }
  box.textContent = 'Consultando…';

  try {
    const r = await withToken('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ questions: list, top_k: 6, prosa })
    });
    const data = await r.json();

    // Construye HTML bonito
    let html = "";
    (data.results || []).forEach((it, idx) => {
      const tag = it.prosa_premium ? '<span class="pill">prosa IA</span>' : '';
      html += `<h3>${idx+1}. ${it.question} ${tag}</h3><div id="ans_${idx}" class="md"></div>`;
      if (Array.isArray(it.sources) && it.sources.length){
        html += `<div class="src"><b>Fuentes:</b> ` +
          it.sources.map(s => `${s.doc_title} (p.${s.page}, score ${s.score})`).join(" · ") +
          `</div>`;
      }
    });
    box.innerHTML = html || "<div class='muted'>Sin resultados.</div>";

    // Render por ítem
    (data.results || []).forEach((it, idx) => {
      const el = document.getElementById('ans_'+idx);
      renderMD(el, it.answer_markdown || "");
    });

  } catch (err) {
    box.textContent = 'ERROR de red: ' + err;
  }
};

// ---------------- REPORTE ----------------
rep.onsubmit = async e => {
  e.preventDefault();
  const periodo = new FormData(rep).get('periodo') || '';
  const repBox = document.getElementById('repres');
  const chartsBox = document.getElementById('repcharts');
  repBox.textContent = 'Generando…';
  chartsBox.innerHTML = '';

  const r = await withToken('/report',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({period: periodo})
  });

  const data = await r.json();
  // Markdown del reporte
  renderMD(repBox, data.report_markdown || JSON.stringify(data,null,2));

  // Gráficos rápidos a partir de KPIs si existen
  const k = data.kpis || {};
  const raw = (k.raw || {});
  const moneyKeys = ['revenue','cogs','gross_profit','operating_income','net_income','ebitda'];
  const ratioKeys = ['gross_margin','operating_margin','net_margin','current_ratio','quick_ratio','debt_ratio','debt_to_equity','ROA','ROE','asset_turnover'];

  const mvals = moneyKeys.filter(k=>raw[k]!=null).map(k=>({k, v: raw[k]}));
  if (mvals.length >= 2){
    const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
    new Chart(cv, {
      type:'bar',
      data:{ labels:mvals.map(x=>x.k), datasets:[{label:'$ (unidades del PDF)', data:mvals.map(x=>x.v), backgroundColor:'#111'}]},
      options:{plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}
    });
  }

  const rvals = ratioKeys.filter(k=>k in kpisFlat(k)).map(k=>({k, v: kpisFlat(k)[k]}));
  if (rvals.length){
    const cv2 = document.createElement('canvas'); chartsBox.appendChild(cv2);
    new Chart(cv2, {
      type:'radar',
      data:{ labels:rvals.map(x=>x.k), datasets:[{label:'Ratios', data:rvals.map(x=>x.v), backgroundColor:'rgba(17,17,17,.15)', borderColor:'#111'}]},
      options:{scales:{r:{beginAtZero:true}}}
    });
  }

  function kpisFlat(k){
    // mezcla top-level + raw
    const flat = Object.assign({}, k, k.raw||{});
    delete flat.raw;
    return flat;
  }
};
</script>
</body>
</html>
"""


# ===================== Rutas públicas =====================
@app.get("/", response_class=HTMLResponse)
async def home():
    html = BASE_HTML.replace("[APPNAME]", APP_NAME_SAFE)
    return HTMLResponse(html)

@app.post("/mp/create-preference")
async def mp_create_preference(payload: Dict[str, str]):
    nombre   = payload.get("nombre", "").strip()
    apellido = payload.get("apellido", "").strip()
    gmail    = payload.get("gmail", "").strip().lower()
    coupon   = (payload.get("coupon") or "").strip().upper()

    if not nombre or not apellido:
        raise HTTPException(400, "Nombre y Apellido son obligatorios")
    if not re.match(r"^[^@\s]+@gmail\.com$", gmail):
        raise HTTPException(400, "Debes usar un correo @gmail.com válido")

    # --- Cupón propio ---
    if coupon == "INVESTU-100":
        # acceso gratis: generar token y NO pasar por Mercado Pago
        uid = make_user_id(gmail)
        ensure_dirs(user_dir(uid))
        token = make_token(uid, 24)
        return {"skip": True, "token": token}

    # (opcional) descuento 50%
    price = float(settings.MP_PRICE_1DAY)
    if coupon == "INVESTU-50":
        price = max(1.0, round(price * 0.5))

    preference = {
        "items": [{
            "title": f"Pase 1 día — {settings.APP_NAME}",
            "quantity": 1,
            "currency_id": settings.MP_CURRENCY,
            "unit_price": price,
        }],
        "payer": {"email": gmail},
        "back_urls": {
            "success": f"{settings.BASE_URL}/mp/return",
            "failure": f"{settings.BASE_URL}/mp/return",
            "pending":  f"{settings.BASE_URL}/mp/return",
        },
        "auto_return": "approved",
        "purpose": "wallet_purchase",
        "metadata": {"gmail": gmail, "nombre": nombre, "apellido": apellido, "coupon": coupon},
        "external_reference": hashlib.sha1(gmail.encode()).hexdigest(),
    }
    try:
        pref = mp.preference().create(preference)["response"]
        return {"id": pref.get("id"),
                "init_point": pref.get("init_point") or pref.get("sandbox_init_point")}
    except Exception as e:
        raise HTTPException(500, f"Mercado Pago error: {e}")


@app.get("/mp/return", response_class=HTMLResponse)
async def mp_return(status: str | None = None, payment_id: str | None = None, collection_id: str | None = None):
    # Si auto_return=approved, vendrá con status=approved y payment_id
    pid = payment_id or collection_id
    if not pid:
        return HTMLResponse("<h3>No se recibió payment_id</h3>", status_code=400)
    try:
        pay = mp.payment().get(pid)["response"]
    except Exception as e:
        return HTMLResponse(f"<h3>Error verificando pago: {e}</h3>", status_code=400)

    st = (pay.get("status") or "").lower()
    payer_email = ((pay.get("payer") or {}).get("email") or "").lower()
    meta = pay.get("metadata") or {}
    gmail_meta = (meta.get("gmail") or "").lower()

    if st != "approved":
        return HTMLResponse(f"<h3>Pago no aprobado (estado: {st}).</h3>")
    if gmail_meta and payer_email and gmail_meta != payer_email:
        return HTMLResponse("<h3 style='color:#b91c1c'>El Gmail no coincide con el pago. Acceso denegado.</h3>")

    gmail = gmail_meta or payer_email or ""
    if not gmail:
        return HTMLResponse("<h3>No se pudo determinar el Gmail del pagador.</h3>")

    uid = make_user_id(gmail)
    ensure_dirs(user_dir(uid))
    token = make_token(uid, 24)
    # Guardamos token en localStorage y vamos al portal
    html = f"<script>localStorage.setItem('token','{token}'); location.href='/portal';</script>"
    return HTMLResponse(html)

@app.get("/portal", response_class=HTMLResponse)
async def portal():
    html = (PORTAL_HTML
            .replace("[MAX_FILES]", str(settings.MAX_UPLOAD_FILES))
            .replace("[SINGLE_MAX]", str(settings.SINGLE_FILE_MAX_MB))
            .replace("[TOTAL_MAX]", str(settings.MAX_TOTAL_MB)))
    return HTMLResponse(html)

# (Opcional) Webhook para notificaciones asincrónicas
# @app.post("/mp/webhook")
# async def mp_webhook(request: Request):
#     body = await request.json()
#     # procesa body['data']['id'] cuando type = payment
#     return {"received": True}

# ===================== Rutas protegidas =====================

def require_user(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "Falta token")
    token = auth.split(" ",1)[1]
    return read_token(token)


def index_worker(base_dir: str, filenames: list[str]):
    """Se ejecuta en segundo plano: abre y agrega PDFs al índice."""
    try:
        base = Path(base_dir)
        pdfs = [base/"docs"/fn for fn in filenames]
        print(f"[index_worker] iniciando, archivos={filenames}", flush=True)
        fragments = add_pdfs_to_index(base, pdfs)
        print(f"[index_worker] listo, fragments_indexed={fragments}", flush=True)
    except Exception as e:
        print(f"[index_worker] ERROR: {e}", flush=True)


@app.post("/upload")
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None),
):
    uid = require_user(request)
    base = user_dir(uid); ensure_dirs(base)

    max_files = settings.MAX_UPLOAD_FILES
    single_max = settings.SINGLE_FILE_MAX_MB * 1024 * 1024
    total_max  = settings.MAX_TOTAL_MB * 1024 * 1024

    if not files or len(files) == 0:
        return {"ok": False, "message": "Selecciona al menos un PDF."}
    if len(files) > max_files:
        return {"ok": False, "message": f"Máximo {max_files} PDFs por subida."}

    total_size = 0
    saved_names: List[str] = []
    for f in files:
        name = (f.filename or "archivo.pdf").strip()
        if not name.lower().endswith(".pdf"):
            return {"ok": False, "message": f"{name}: solo se aceptan PDF"}
        content = await f.read()                     # SOLO guardamos; nada pesado aquí
        total_size += len(content)
        if len(content) > single_max:
            return {"ok": False, "message": f"{name}: supera {settings.SINGLE_FILE_MAX_MB} MB"}
        with open(base/"docs"/name, "wb") as out:
            out.write(content)
        saved_names.append(name)

    if total_size > total_max:
        for nm in saved_names:
            try: (base/"docs"/nm).unlink(missing_ok=True)
            except: pass
        return {"ok": False, "message": f"Superaste el total permitido ({settings.MAX_TOTAL_MB} MB por subida)."}

    # 👉 La indexación pesada se hace en segundo plano (evita el 502/504)
    background_tasks.add_task(index_worker, str(base), saved_names)

    return {
        "ok": True,
        "saved": saved_names,
        "errors": [],
        "indexing": "in_progress",
        "note": "Estamos procesando tus PDFs en segundo plano."
    }


@app.get("/status")
def status(request: Request):
    uid = require_user(request)
    base = user_dir(uid)
    idx, meta = load_index(base)
    return {"fragments": idx.ntotal, "docs": len({m['doc_title'] for m in meta})}


import re as _re2

def _mk_sources(res, n=5):
    out = []
    for s, m in res[:n]:
        out.append({"doc_title": m["doc_title"], "page": m["page"], "score": round(float(s), 2)})
    return out

def summarize_answer(question: str, res) -> str:
    """
    Resumen heurístico en español a partir de los mejores pasajes.
    No usa APIs externas.
    """
    # Une los 6 mejores trozos de texto
    texts = [m["text"] for _, m in res[:6] if "text" in m]
    blob = " ".join(texts)

    # Oraciones
    sents = _re2.split(r"(?<=[\.\!\?])\s+", blob)
    # Palabras gatillo típicas de informe financiero
    kws = ["Aumente", "Reduzca", "Mejore", "Optimice", "Cuidado",
           "Liquidez", "Endeudamiento", "Rentabilidad", "EBITDA",
           "Margen", "Inventario", "Cuentas por cobrar", "Cuentas por pagar",
           "Capital de Trabajo", "Equilibrio", "Tesorería"]

    bullets = []
    for kw in kws:
        for s in sents:
            if kw.lower() in s.lower():
                bullets.append(s.strip())
    # Quita duplicados y limita
    seen = set(); clean = []
    for b in bullets:
        if b not in seen:
            seen.add(b); clean.append(b)
    if not clean:
        # si no detectó recomendaciones, toma 3 frases representativas
        clean = [s.strip() for s in sents[:3]]

    # Arma respuesta
    out = []
    out.append("### Resumen")
    for b in clean[:8]:
        out.append(f"- {b}")

    # Señales rápidas (si aparecen números)
    def grab(rx):
        m = _re2.search(rx, blob, flags=_re2.IGNORECASE)
        return m.group(0) if m else None

    ebitda = grab(r"EBITDA[^0-9\-]*[-\$\s]*[\d\.\,]+")
    margen = grab(r"Margen\s+\w+[^0-9\-]*[-\%\s]*[\d\.\,]+%")
    deuda  = grab(r"(Deuda/Activos|Deuda/Patrimonio|Ratio de Endeudamiento)[^%]*\d[\d\.\,]*%?")
    if any([ebitda, margen, deuda]):
        out.append("\n### Indicadores detectados")
        if ebitda: out.append(f"- {ebitda}")
        if margen: out.append(f"- {margen}")
        if deuda:  out.append(f"- {deuda}")

    # Recomendaciones (heurística)
    recs = [s for s in sents if any(w in s.lower() for w in ["aument", "reduzc", "mejor", "optim", "equilibr"])]
    if recs:
        out.append("\n### Recomendaciones")
        for r in recs[:6]:
            out.append(f"- {r.strip()}")

    return "\n".join(out)


@app.post("/ask")
async def ask(request: Request, body: Dict[str, Any]):
    uid = require_user(request)
    base = user_dir(uid)
    top_k = int(body.get("top_k", settings.TOP_K))
    prosa = bool(body.get("prosa", False))

    q = body.get("questions") or body.get("question") or []
    if isinstance(q, str):
        # admite varias líneas en una caja de texto
        queries = [x.strip() for x in q.splitlines() if x.strip()]
    else:
        queries = [str(x).strip() for x in q][:5]

    if not queries:
        raise HTTPException(400, "Falta 'questions' o 'question'")

    results = []
    for query in queries:
        # Busca pasajes relevantes
        res = semantic_search(base, query, top_k)  # [(score, meta), ...]

        # Preparamos el contexto para ambas modalidades
        ctx = [{"score": round(float(s), 3), **m} for s, m in res]

        # Si el usuario activó "prosa" y hay API key → usa Prosa Premium
        if prosa and _premium_on():
            answer = premium_answer_for_question(query, ctx)
        else:
            # Resumen heurístico local (sin GPT)
            answer = summarize_answer(query, res)

        sources = _mk_sources(res, n=5)

        results.append({
            "question": query,
            "answer_markdown": answer,   # siempre devolvemos Markdown
            "sources": sources,
            "prosa_premium": bool(prosa and _premium_on())
        })

    return {"results": results}


# ---- KPI extractor (igual que antes, simplificado) ----
import re as _re
LABELS = {
    "revenue": ["ingresos","ventas","ventas netas","ventas totales"],
    "cogs": ["costo de ventas","coste de ventas","costo de los bienes vendidos"],
    "gross_profit": ["utilidad bruta","resultado bruto"],
    "operating_income": ["resultado operacional","utilidad de operación","resultado de explotación","ebit"],
    "net_income": ["utilidad neta","resultado del ejercicio","ganancia neta"],
    "total_assets": ["activos totales","total activos"],
    "total_equity": ["patrimonio","patrimonio neto","capital contable"],
    "total_liabilities": ["pasivos totales","total pasivos","deudas totales"],
    "current_assets": ["activos corrientes","activos circulantes"],
    "current_liabilities": ["pasivos corrientes","pasivos de corto plazo"],
    "inventory": ["inventario","existencias"],
    "accounts_receivable": ["cuentas por cobrar","deudores comerciales"],
    "accounts_payable": ["cuentas por pagar","acreedores comerciales"],
    "interest_expense": ["gastos financieros","intereses pagados"],
    "ebitda": ["ebitda"],
}

NUM = _re.compile(r"[-+]?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?")

def _num(s: str):
    s = s.replace(" ",""); m = NUM.search(s)
    if not m: return None
    n = m.group(0)
    if n.count(",")>0 and n.count(".")==0: n = n.replace(".","").replace(",",".")
    else: n = n.replace(",","")
    try: return float(n)
    except: return None

def extract_kpis_from_pdfs(pdfs: List[Path]) -> Dict[str, Any]:
    import pdfplumber
    values: Dict[str, float] = {}
    for pdf in pdfs:
        try:
            with pdfplumber.open(pdf) as doc:
                for page in doc.pages:
                    text = page.extract_text() or ""
                    low = text.lower()
                    for key, keys in LABELS.items():
                        for k in keys:
                            if k in low:
                                for ln in text.splitlines():
                                    if k in ln.lower():
                                        v = _num(ln)
                                        if v is not None: values.setdefault(key, v)
        except Exception:
            continue

    out: Dict[str, Any] = {"raw": values}

    # Conveniencias
    rev   = values.get("revenue")
    cogs  = values.get("cogs")
    gp    = values.get("gross_profit")
    op    = values.get("operating_income")
    ni    = values.get("net_income")
    assets= values.get("total_assets")
    eq    = values.get("total_equity")
    liab  = values.get("total_liabilities")
    ca    = values.get("current_assets")
    cl    = values.get("current_liabilities")
    inv   = values.get("inventory")
    ar    = values.get("accounts_receivable")
    ap    = values.get("accounts_payable")
    intexp= values.get("interest_expense")
    ebitda= values.get("ebitda")

    # Márgenes
    if rev and cogs: out["gross_margin"]=(rev-cogs)/rev
    if op and rev:   out["operating_margin"]=op/rev
    if ni and rev:   out["net_margin"]=ni/rev

    # Liquidez
    if ca and cl and cl!=0: out["current_ratio"]=ca/cl
    if ca and inv is not None and cl and cl!=0: out["quick_ratio"]=(ca - inv)/cl

    # Endeudamiento
    if liab and assets and assets!=0: out["debt_ratio"]=liab/assets
    if liab and eq and eq!=0:         out["debt_to_equity"]=liab/eq
    if op and intexp and intexp!=0:   out["interest_coverage"]=op/intexp

    # Rentabilidad
    if ni and assets and assets!=0: out["ROA"]=ni/assets
    if ni and eq and eq!=0:         out["ROE"]=ni/eq
    if rev and assets and assets!=0: out["asset_turnover"]=rev/assets

    # Actividad (días)
    if rev and ar:    out["days_receivable"]=365*ar/rev
    if cogs and ap:   out["days_payable"]=365*ap/cogs
    if cogs and inv:  out["inventory_turnover"]=cogs/max(inv,1e-9)

    # Working capital
    if ca is not None and cl is not None: out["working_capital"]=ca - cl

    return out


@app.post("/report")
async def report(request: Request, body: Dict[str, Any]):
    uid = require_user(request)
    base = user_dir(uid)
    pdfs = list((base/"docs").glob("*.pdf"))
    if not pdfs:
        return {"message":"Sube al menos un PDF"}

    period = (body.get("period") or "").strip()
    k = extract_kpis_from_pdfs(pdfs)

    def pct(x): return f"{x*100:.1f}%" if x is not None else "s/d"
    val = lambda x: f"{x:,.0f}".replace(",",".") if isinstance(x,(int,float)) else "s/d"

    lines = []
    lines.append(f"**Período**: {period or 'no especificado'}")
    lines.append("")
    lines.append("## 1) Resumen de KPIs")
    for key in ["revenue","cogs","gross_profit","operating_income","net_income","total_assets","total_equity","total_liabilities","current_assets","current_liabilities","inventory","accounts_receivable","accounts_payable","ebitda","interest_expense","working_capital"]:
        if key in k.get("raw",{}):
            lines.append(f"- {key}: {val(k['raw'][key])}")

    lines.append("")
    lines.append("## 2) Márgenes")
    lines.append(f"- Margen Bruto: {pct(k.get('gross_margin'))}")
    lines.append(f"- Margen Operacional: {pct(k.get('operating_margin'))}")
    lines.append(f"- Margen Neto: {pct(k.get('net_margin'))}")

    lines.append("")
    lines.append("## 3) Liquidez")
    lines.append(f"- Razón Corriente (CA/CL): {k.get('current_ratio','s/d')}")
    lines.append(f"- Prueba Ácida (CA-Inv)/CL: {k.get('quick_ratio','s/d')}")

    lines.append("")
    lines.append("## 4) Endeudamiento")
    lines.append(f"- Deuda/Activos: {pct(k.get('debt_ratio')) if isinstance(k.get('debt_ratio'),float) else k.get('debt_ratio','s/d')}")
    lines.append(f"- Deuda/Patrimonio: {k.get('debt_to_equity','s/d')}")
    lines.append(f"- Cobertura de Intereses (EBIT/Intereses): {k.get('interest_coverage','s/d')}")

    lines.append("")
    lines.append("## 5) Rentabilidad")
    lines.append(f"- ROA: {pct(k.get('ROA'))}")
    lines.append(f"- ROE: {pct(k.get('ROE'))}")
    lines.append(f"- Rotación de Activos (Ventas/Activos): {k.get('asset_turnover','s/d')}")

    lines.append("")
    lines.append("## 6) Actividad")
    lines.append(f"- Días de Cuentas por Cobrar: {k.get('days_receivable','s/d')}")
    lines.append(f"- Días de Cuentas por Pagar: {k.get('days_payable','s/d')}")
    lines.append(f"- Rotación de Inventario (COGS/Inv): {k.get('inventory_turnover','s/d')}")

    lines.append("")
    lines.append("## 7) Flujo de Caja")
    lines.append("- Requiere estado de flujos para detalle. Si lo tienes, súbelo para estimar CFO/CFI/CFF y cobertura de caja.")

    lines.append("")
    lines.append("## 8) Punto de Equilibrio")
    lines.append("- Se calcula con costos fijos y margen de contribución. Si el PDF expone ambos, puedo estimarlo en una versión futura.")

    lines.append("")
    lines.append("## 9) Análisis Vertical y Horizontal")
    lines.append("- Con balances comparativos y resultados por períodos puedo generar AV/AH. Sube estados con al menos 2 años.")

    lines.append("")
    lines.append("## 10) Apalancamiento (F, O, T)")
    lines.append("- Con detalle de costos fijos/variables y estructura de capital puedo calcular grados de apalancamiento.")

    lines.append("")
    lines.append("## 11) Modelo Z (quiebra)")
    lines.append("- Requiere activo circulante, pasivo circulante, utilidades retenidas, EBIT, valor de mercado del patrimonio y ventas.")
    lines.append("- Si subes esos datos (o estados detallados), puedo estimarlo.")

    lines.append("")
    lines.append("## 12) Tesorería (30/60/90)")
    lines.append("- Con antigüedad de saldos de clientes/proveedores, puedo construir el semáforo a 30/60/90.")

    lines.append("")
    lines.append("## Conclusiones & Recomendaciones")
    tips = []
    if k.get("current_ratio") and k["current_ratio"]<1:
        tips.append("Refuerza capital de trabajo: mejora cobros, renegocia plazos con proveedores, reduce inventario lento.")
    if k.get("debt_to_equity") and k["debt_to_equity"]>2:
        tips.append("Alto apalancamiento: evalúa capitalizar utilidades o reestructurar deuda para bajar riesgo financiero.")
    if k.get("net_margin") and k["net_margin"]<0.05:
        tips.append("Margen neto bajo: revisa gastos fijos y precios; busca eficiencias operativas.")
    if not tips: tips.append("Los datos son parciales; sube estados más detallados para un diagnóstico profundo.")
    lines += [f"- {t}" for t in tips]

    # ----- Resumen ejecutivo (opcional con GPT) -----
    executive_summary = premium_exec_summary(period, k)  # devuelve None si no hay OPENAI_API_KEY

    result_md = "\n".join(lines)
    if executive_summary:
        result_md = "## Resumen Ejecutivo (IA)\n" + executive_summary + "\n\n" + result_md

    return {
        "kpis": k,
        "report_markdown": result_md,
        "executive_summary": executive_summary
    }



# Salud

@app.get("/__check_coupon")
def __check_coupon():
    return {
        "coupon_field_in_template": "name=coupon" in BASE_HTML
    }


@app.get("/health")
async def health():
    return PlainTextResponse("ok", status_code=200)

@app.get("/__version")
def __version():
    import re
    # intenta extraer cualquier correo que aparezca en el HTML
    m = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+', BASE_HTML)
    email = m.group(0) if m else None
    return {
        "has_coupon_field": 'name="coupon"' in BASE_HTML,
        "email_in_html": email,
        "has_dreamingup7": 'dreamingup7@gmail.com' in BASE_HTML,
    }
