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
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from jose import jwt, JWTError
from zipfile import ZipFile
import tempfile
import aiohttp

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

# CORS: admitir apex y www + localhost
allowed = {
    "http://localhost:8000", "http://127.0.0.1:8000",
    "https://inbestu.com", "https://www.inbestu.com",
    "https://vedetodo.online", "https://www.vedetodo.online",
}
# También la BASE_URL exacta por si cambia
if settings and getattr(settings, "BASE_URL", None):
    allowed.add(settings.BASE_URL)
    # variante http/https por si toca
    if settings.BASE_URL.startswith("https://"):
        allowed.add(settings.BASE_URL.replace("https://","http://"))
    if settings.BASE_URL.startswith("http://"):
        allowed.add(settings.BASE_URL.replace("http://","https://"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(allowed),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    "INVESTU-100": {"type": "free", "desc": "Acceso gratis"},  # ← cambia aquí
    "INVESTU-50":  {"type": "percent", "value": 50, "desc": "50% OFF"},
    "PASE-GRATIS": {"type": "free", "desc": "Gratis 1 día"},
}

# Guardas de configuración para MP
def _require_mp():
    if not settings:
        raise HTTPException(500, f"Config faltante: {SETTINGS_ERROR or 'sin settings'}")
    if not getattr(settings, "MP_ACCESS_TOKEN", None):
        raise HTTPException(500, "MP_ACCESS_TOKEN no configurado")
    if mp is None:
        raise HTTPException(500, "SDK de Mercado Pago no inicializado")



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
    import faiss
    p = idx_paths(base)
    dim = get_model().get_sentence_embedding_dimension()
    if p["faiss"].exists() and p["meta"].exists():
        idx = faiss.read_index(str(p["faiss"]))
        meta = []
        with open(p["meta"], "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: 
                    continue
                try:
                    meta.append(json.loads(line))
                except Exception:
                    # ignora línea parcial si justo se leyó durante escritura
                    continue
        return idx, meta
    return faiss.IndexFlatIP(dim), []


def save_index(base: Path, idx, meta: List[Dict[str, Any]]):
    import faiss, os  # import aquí
    p = idx_paths(base)
    p["faiss"].parent.mkdir(parents=True, exist_ok=True)

    # 1) FAISS atómico
    tmp_faiss = p["faiss"].with_suffix(p["faiss"].suffix + ".tmp")
    faiss.write_index(idx, str(tmp_faiss))
    os.replace(tmp_faiss, p["faiss"])

    # 2) metadata.jsonl atómico
    tmp_meta = p["meta"].with_suffix(p["meta"].suffix + ".tmp")
    with open(tmp_meta, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(tmp_meta, p["meta"])


def add_pdfs_to_index(base: Path, pdfs: List[Path]) -> int:
    import fitz  # PyMuPDF
    ensure_dirs(base)
    idx, meta = load_index(base)

    new_txt, new_meta = [], []

    for pdf in pdfs:
        if not pdf.exists():
            continue
        try:
            with fitz.open(pdf) as doc:
                for page_num, page in enumerate(doc, start=1):
                    # 1) Texto "normal"
                    text = clean_text(page.get_text("text") or "")

                    # 2) Fallback por bloques (capta tablas/cabeceras en varios PDFs)
                    if len(text) < 50:
                        try:
                            blocks = page.get_text("blocks") or []
                            blk_txt = " ".join(
                                b[4] for b in blocks
                                if isinstance(b, (list, tuple)) and len(b) > 4 and isinstance(b[4], str)
                            )
                            text = clean_text(blk_txt or text)
                        except Exception:
                            pass

                    # 3) Fallback pdfplumber (último recurso)
                    if len(text) < 50:
                        try:
                            import pdfplumber
                            with pdfplumber.open(str(pdf)) as pp:
                                pn = page_num - 1
                                if 0 <= pn < len(pp.pages):
                                    text2 = pp.pages[pn].extract_text() or ""
                                    text = clean_text(text2) or text
                        except Exception:
                            pass

                    if len(text) < 50:
                        # página sin texto útil (escaneada o imagen)
                        continue

                    for ck in chunk_text(text):
                        new_txt.append(ck)
                        new_meta.append({
                            "doc_title": pdf.name,
                            "page": page_num,
                            "text": ck
                        })
        except Exception as e:
            print(f"[add_pdfs_to_index] ERROR abriendo {pdf}: {e}", flush=True)
            continue

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

import secrets

def _session_code(n: int = 6) -> str:
    # Código amigable (evita 0/O y 1/I)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(n))

def make_token(uid: str, hours: int = 24) -> str:
    scode = _session_code()
    payload = {
        "sub": uid,
        "code": scode,  # ← código visible para el usuario
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=hours)).timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

def read_token_full(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

def _get_token_from_request(request: Request) -> str | None:
    # 1) Authorization: Bearer xxx
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1]
    # 2) Cookie: token=xxx
    cookie = request.headers.get("cookie") or request.headers.get("Cookie") or ""
    for part in cookie.split(";"):
        k, _, v = part.strip().partition("=")
        if k == "token" and v:
            return v
    return None

def read_token(token: str) -> str:
    data = read_token_full(token)
    return data["sub"]

def require_user(request: Request) -> str:
    token = _get_token_from_request(request)
    if not token:
        raise HTTPException(401, "Falta token")
    return read_token(token)


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
    Redacción clara con DEFINICIONES + DIAGNÓSTICO + RECOMENDACIONES.
    Usa solo la evidencia (ctx). Devuelve Markdown.
    Si la llamada a GPT falla, cae a diagnóstico local.
    """
    # evidencia (top-6)
    snips = []
    for c in (ctx or [])[:6]:
        doc = c.get("doc_title","?"); pg = c.get("page","?"); tx = c.get("text","")
        snips.append(f"[{doc} p.{pg}] {tx}")
    evidence = "\n\n".join(snips) or "(sin evidencia)"

    sys = (
        "Eres un analista financiero senior que escribe en español. "
        "Estructura SIEMPRE tu salida con estos apartados: "
        "1) **Definiciones clave** (explica brevemente cada concepto mencionado en la pregunta o la evidencia: liquidez, endeudamiento, margen, EBITDA, etc.). "
        "2) **Diagnóstico** (evalúa si los factores son buenos/regulares/malos; justifica con umbrales típicos o comparaciones relativas a la evidencia; sé prudente si faltan datos). "
        "3) **Recomendaciones** (acciones concretas, priorizadas y medibles; separa corto/mediano plazo). "
        "4) **Riesgos/alertas** (señales que vigilar). "
        "Usa Markdown, tono claro, sin inventar cifras fuera de la evidencia."
    )
    usr = (
        f"Pregunta:\n{question}\n\n"
        f"Evidencia permitida (usa SOLO esto):\n{evidence}\n\n"
        "Formato:\n"
        "### Definiciones clave\n"
        "- ...\n\n"
        "### Diagnóstico\n"
        "- ...\n\n"
        "### Recomendaciones (prioridad)\n"
        "- [P1] ...\n- [P2] ...\n\n"
        "### Riesgos / alertas\n"
        "- ...\n"
    )
    # Llamada a GPT
    if not _premium_on():
        # si no hay API key, usa la vía local
        return rule_based_advice_from_ctx(question, ctx) or "No hay suficientes datos para responder."
    try:
        out = _gpt(sys, usr, max_tokens=900)
        if out: 
            return out
        # si vino vacío, usa fallback local
        return rule_based_advice_from_ctx(question, ctx) or "No hay suficientes datos para responder."
    except Exception:
        return rule_based_advice_from_ctx(question, ctx) or "No hay suficientes datos para responder."


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
  localStorage.setItem('token', data.token);
  const secure = (location.protocol === 'https:') ? '; Secure' : '';
  document.cookie = 'token=' + data.token + '; Path=/; Max-Age=86400; SameSite=Lax' + secure;
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
  <p class="muted">Acceso por 24h tras el pago con Mercado Pago. Sube tus informes PDF y obtén KPI + análisis + sugerencias. Para soporte: <b>dreamingup7@gmail.com</b>.</p>

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
  .pill{display:inline-block;font-size:.75rem;color:#111;background:#e5e7eb;border-radius:999px;padding:.2rem .6rem;margin-left:.4rem;white-space:nowrap}
  .src{font-size:.85rem;color:var(--muted)}
  .charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:10px}
  .kpi{display:grid;grid-template-columns:1fr auto;gap:6px;padding:.6rem .8rem;border:1px solid #eee;border-radius:10px;background:#fbfbfb}
  .bar{display:flex; gap:8px; align-items:center; flex-wrap:wrap}
  .nowrap{white-space:nowrap}
  @media print {
    body{background:#fff}
    .card{box-shadow:none;border:none}
    .no-print{display:none !important}
  }
</style>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.9/dist/purify.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="card">
  <div class="bar no-print">
    <h2 style="margin:0;flex:1 1 auto">Portal de usuario (pase activo)</h2>
    <button onclick="window.print()">🖨️ Imprimir reporte</button>
  </div>
  <div id="sessionbar" class="muted" style="margin:.4rem 0"></div>

  <!-- SUBIR PDFs -->
  <form id="up" enctype="multipart/form-data" class="no-print" onsubmit="return false;">
    <input id="fileInput" type="file" name="files" multiple accept="application/pdf">
    <small class="muted">Límites: máx <b>[MAX_FILES]</b> PDFs por subida · <b>[SINGLE_MAX] MB</b> cada uno · hasta <b>[TOTAL_MAX] MB</b> en total.</small>
    <div class="bar" style="margin-top:8px">
      <button id="btnUpload" type="button" onclick="doUpload()">Subir PDFs e indexar</button>
      <small class="muted">También puedes seleccionar 1 archivo, subirlo, y repetir para agregar “uno a uno”.</small>
    </div>
  </form>
  <pre id="upres"></pre>

  <!-- PREGUNTAS (hasta 5 a la vez) -->
  <form id="askf" class="no-print">
    <textarea name="questions" rows="4" placeholder="Escribe hasta 5 preguntas, una por línea"></textarea>
    <label class="nowrap" style="display:inline-flex;gap:6px;align-items:center;margin-top:8px">
      <input type="checkbox" id="prosa"> Prosa premium (IA)
    </label>
    <div style="margin-top:8px"><button type="submit">Preguntar</button></div>
  </form>
  <div class="muted" style="margin:.4rem 0">Puedes hacer hasta 5 preguntas a la vez (una por línea).</div>
  <div id="askres" class="md"></div>

  <hr class="no-print" style="margin:18px 0">

  <!-- REPORTE -->
  <form id="rep" class="no-print">
    <input name="periodo" placeholder="Período (opcional)">
    <small class="muted">Indica meses o años a evaluar. Ej: “2022–2024”, “ene–jun 2024”, “últimos 12 meses”.</small>
    <div style="margin-top:8px"><button>Generar Reporte Automático</button></div>
  </form>

  <div id="repres" class="md" style="margin-top:10px"></div>
  <div id="repcharts" class="charts"></div>
</div>

<script>
const MAX_FILES = [MAX_FILES];
const SINGLE_MAX = [SINGLE_MAX]; // MB por archivo
const TOTAL_MAX  = [TOTAL_MAX];  // MB por subida
const up   = document.getElementById('up');
const askf = document.getElementById('askf');
const rep  = document.getElementById('rep');


// --- Verificación de sesión al cargar ---
(async function checkSession(){
  try{
    const r = await withToken('/me');
    if(r.status!==200){ location.href='/'; return; }
    const me = await r.json();
    // Código + cuenta regresiva
    const sb = document.getElementById('sessionbar');
    const render = ()=> {
      const mins = Math.floor(me.remaining_seconds/60);
      const secs = me.remaining_seconds%60;
      sb.textContent = `Código de sesión: ${me.code} · expira en ${mins}:${secs.toString().padStart(2,'0')}`;
      if(me.remaining_seconds>0){ me.remaining_seconds--; setTimeout(render,1000); }
      else { sb.textContent = "Sesión expirada. Vuelve a pagar o usar tu cupón." }
    };
    render();
  }catch(e){ location.href='/'; }
})();

function toMB(n){ return (n/1024/1024).toFixed(1) + " MB"; }

async function withToken(url, opts = {}) {
  const t = localStorage.getItem('token') || '';
  opts.credentials = 'include';
  // 👉 Siempre pedimos JSON para que el backend no intente devolver HTML
  const baseHeaders = { Accept: 'application/json' };
  if (t) baseHeaders.Authorization = 'Bearer ' + t;
  opts.headers = { ...baseHeaders, ...(opts.headers || {}) };
  return fetch(url, opts);
}


// ------ Render Markdown + Matemáticas + Gráficos ------
function renderMD(el, md){
  try {
    const raw = marked.parse(md || "");
    const clean = DOMPurify.sanitize(raw);
    el.innerHTML = clean;
    try { renderMathInElement(el,{ delimiters:[
      {left:"$$",right:"$$",display:true},{left:"$",right:"$",display:false}
    ]}); } catch(e){}
    // ```chart
    const blocks = el.querySelectorAll("pre code.language-chart");
    blocks.forEach((code) => {
      let cfg = null; try{ cfg = JSON.parse(code.textContent.trim()); }catch(e){ return; }
      const pre = code.closest("pre");
      const canvas = document.createElement("canvas");
      pre.replaceWith(canvas);
      new Chart(canvas.getContext('2d'), cfg);
    });
  } catch(err) { el.textContent = "Error renderizando Markdown: " + err; }
}


// --- Diagnóstico mínimo (muestra errores JS en pantalla) ---
window.addEventListener('error', (e) => {
  const box = document.getElementById('upres');
  if (box) box.textContent = 'Error JS: ' + (e.message || (e.error && e.error.message) || 'desconocido');
});
document.getElementById('upres').textContent = 'JS cargó correctamente.';


// ---------------- SUBIR PDFs ----------------

// Elementos
const fi    = document.getElementById('fileInput');
const upBox = document.getElementById('upres');
const btnU  = document.getElementById('btnUpload');

// Limites con fallback por si algo no se reemplazó
const LIMITS = {
  max:   Number(typeof MAX_FILES  !== 'undefined' ? MAX_FILES  : 5),
  single:Number(typeof SINGLE_MAX !== 'undefined' ? SINGLE_MAX : 20),  // MB
  total: Number(typeof TOTAL_MAX  !== 'undefined' ? TOTAL_MAX  : 100), // MB
};

// Validador común
function validateFiles(files){
  if (!files.length)         return { ok:false, msg:'Selecciona al menos un PDF.' };
  if (files.length > LIMITS.max)
                             return { ok:false, msg:`Máximo ${LIMITS.max} PDFs por subida.` };
  let total = 0;
  for (const f of files){
    total += f.size;
    if (!/\.pdf$/i.test(f.name))                   return { ok:false, msg:`${f.name}: solo PDF` };
    if (f.size > LIMITS.single*1024*1024)          return { ok:false, msg:`${f.name} supera ${LIMITS.single} MB (${toMB(f.size)})` };
  }
  if (total > LIMITS.total*1024*1024)
    return { ok:false, msg:`Superaste el total permitido (${LIMITS.total} MB). Estás subiendo ${toMB(total)}.` };
  return { ok:true };
}

// Sólo validar al elegir (no sube todavía)
fi?.addEventListener('change', () => {
  const files = Array.from(fi.files || []);
  const v = validateFiles(files);
  if (!v.ok){ upBox.textContent = v.msg; fi.value = ''; return; }
  upBox.textContent = `Listo: ${files.map(f=>f.name).join(', ')}. Ahora presiona “Subir PDFs e indexar”.`;
});

// Subir lo seleccionado
async function doUpload(){
  const files = Array.from(fi?.files || []);
  const v = validateFiles(files);
  if (!v.ok){
    upBox.textContent = v.msg;
    if (!files.length) fi?.click(); // abre selector si no hay archivos
    return;
  }

  const fd = new FormData();
  files.forEach(f => fd.append('files', f));

  upBox.textContent = 'Subiendo…';
  try{
    const r   = await withToken('/upload', { method:'POST', body: fd });
    const txt = await r.text();
    let data; try{ data = JSON.parse(txt); } catch{ data = null; }

    if (!r.ok || (data && data.ok === false)){
      const msg = (data && (data.message || data.detail)) || txt || `Error HTTP ${r.status}`;
      upBox.textContent = `Error subiendo archivos: ${msg}`;
      return;
    }

    const names = (data?.saved || []).join(', ');
    upBox.textContent = names
      ? `Cargados: ${names}. Se están procesando en segundo plano.`
      : 'Subida completa. Se están procesando en segundo plano.';
    fi.value = ''; // limpiar selección

    // ✅ NUEVO: consultar el estado del índice y mostrarlo
    try {
      const s = await withToken('/status');
      if (s.ok) {
        const st = await s.json();
        upBox.textContent += `\nÍndice: ${st.fragments} fragmentos · ${st.docs} documento(s).`;
      }
    } catch (e) {
      // si falla, lo ignoramos silenciosamente
    }

  }catch(err){
    upBox.textContent = 'Error de red o sesión: ' + err;
  }
}

// Botón y Enter del formulario
document.getElementById('up')?.addEventListener('submit', (e) => { e.preventDefault(); doUpload(); });



// ---------------- PREGUNTAR ----------------
document.getElementById('askf')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const raw = new FormData(document.getElementById('askf')).get('questions') || '';
  const list = raw.split(/\r?\n/).map(s => s.trim()).filter(Boolean).slice(0, 5);
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

    let html = "";
    (data.results || []).forEach((it, idx) => {
      const tag = it.prosa_premium ? '<span class="pill">Prosa premium (IA)</span>' : '';
      html += `<h3>${idx+1}. ${it.question} ${tag}</h3><div id="ans_${idx}" class="md"></div>`;
      if (Array.isArray(it.sources) && it.sources.length){
        html += `<div class="src"><b>Fuentes:</b> ` +
          it.sources.map(s => `${s.doc_title} (p.${s.page}, score ${s.score})`).join(" · ") +
          `</div>`;
      }
    });
    box.innerHTML = DOMPurify.sanitize(html || "<div class='muted'>Sin resultados.</div>");
    (data.results || []).forEach((it, idx) => renderMD(document.getElementById('ans_'+idx), it.answer_markdown || ""));
  } catch (err) {
    box.textContent = 'ERROR de red: ' + err;
  }
});


// ---------------- REPORTE ----------------

document.getElementById('rep')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const periodo = new FormData(document.getElementById('rep')).get('periodo') || '';
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
  renderMD(repBox, data.report_markdown || JSON.stringify(data,null,2));


  // --- Datos base para todos los gráficos ---
  const k = data.kpis || {};
  const raw = (k.raw || {});
  const byYear = (k.by_year || {});
  const names = {
    revenue:["Ingresos","Revenue"],
    cogs:["Costo de ventas","COGS"],
    gross_profit:["Utilidad bruta","Gross Profit"],
    operating_income:["Resultado operacional (EBIT)","Operating Income"],
    net_income:["Utilidad neta","Net Income"],
    ebitda:["EBITDA","EBITDA"],
    current_ratio:["Razón Corriente","Current Ratio"],
    quick_ratio:["Prueba Ácida","Quick Ratio"],
    gross_margin:["Margen Bruto","Gross Margin"],
    operating_margin:["Margen Operacional","Operating Margin"],
    net_margin:["Margen Neto","Net Margin"],
    debt_ratio:["Deuda/Activos","Debt Ratio"],
    debt_to_equity:["Deuda/Patrimonio","Debt to Equity"],
    ROA:["ROA","Return on Assets"],
    ROE:["ROE","Return on Equity"],
    asset_turnover:["Rotación de Activos","Asset Turnover"]
  };

  function band(val, low, high, reverse=false){
    if(val==null||isNaN(val)) return {label:"s/d", color:"#9ca3af"};
    if(reverse){
      if(high!=null && val>high) return {label:"Riesgo", color:"#ef4444"};
      if(low!=null  && val>low)  return {label:"Medio", color:"#f59e0b"};
      return {label:"Bueno", color:"#10b981"};
    }else{
      if(low!=null && val<low)   return {label:"Riesgo", color:"#ef4444"};
      if(high!=null && val<high) return {label:"Medio", color:"#f59e0b"};
      return {label:"Bueno", color:"#10b981"};
    }
  }
  function kpisFlat(kobj){
    const flat = Object.assign({}, kobj, kobj.raw||{});
    delete flat.raw; return flat;
  }

  // 0) Tarjetas semáforo (didáctico)
  (() => {
    const flat = kpisFlat(k);
    const thresholds = {
      current_ratio:{low:1.0,high:1.5,reverse:false},
      quick_ratio:{low:0.8,high:1.0,reverse:false},
      gross_margin:{low:0.20,high:0.35,reverse:false},
      operating_margin:{low:0.05,high:0.15,reverse:false},
      net_margin:{low:0.03,high:0.10,reverse:false},
      debt_ratio:{low:0.50,high:0.70,reverse:true},
      debt_to_equity:{low:1.5,high:2.5,reverse:true},
      interest_coverage:{low:1.5,high:3.0,reverse:false},
      ROA:{low:0.03,high:0.07,reverse:false},
      ROE:{low:0.10,high:0.20,reverse:false},
      asset_turnover:{low:0.60,high:1.00,reverse:false},
    };
    const explain = {
      current_ratio:"Liquidez de corto plazo (ideal ≥ 1,5).",
      quick_ratio:"Liquidez exigente sin inventario (ideal ≥ 1,0).",
      gross_margin:"Utilidad tras costo directo.",
      operating_margin:"Eficiencia operativa.",
      net_margin:"Utilidad final.",
      debt_ratio:"Deuda sobre activos (bajo es mejor).",
      debt_to_equity:"Deuda por cada peso de patrimonio.",
      interest_coverage:"Veces que el EBIT cubre intereses.",
      ROA:"Rentabilidad de los activos.",
      ROE:"Rentabilidad del patrimonio.",
      asset_turnover:"Eficiencia comercial vs activos."
    };
    const keys = Object.keys(thresholds).filter(k => flat[k] != null);
    if(!keys.length) return;
    const panel = document.createElement('div');
    panel.style.display='grid';
    panel.style.gridTemplateColumns='repeat(auto-fit,minmax(240px,1fr))';
    panel.style.gap='12px';
    keys.forEach(key=>{
      const t = thresholds[key];
      const b = band(flat[key], t.low, t.high, !!t.reverse);
      const card = document.createElement('div');
      card.className='kpi';
      card.innerHTML = `
        <div>
          <div><b>${names[key][0]} / ${names[key][1]}</b></div>
          <div class="muted">${explain[key]}</div>
        </div>
        <div class="pill" style="color:#fff;background:${b.color}">${b.label}</div>
      `;
      panel.appendChild(card);
    });
    chartsBox.appendChild(panel);
  })();

  // 1) Barras dinero (ES/EN)
  (() => {
    const keys = ['revenue','operating_income','net_income','ebitda'].filter(k=>raw[k]!=null);
    if(keys.length<2) return;
    const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
    new Chart(cv,{type:'bar',
      data:{labels:keys.map(k=>names[k][0]+' / '+names[k][1]),
            datasets:[{label:'$ (unidades del PDF)', data:keys.map(k=>raw[k]), backgroundColor:'#111'}]},
      options:{plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}
    });
  })();

  // 2) Donut composición
  (() => {
    if(raw.revenue!=null && raw.cogs!=null && raw.gross_profit!=null){
      const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
      new Chart(cv,{type:'doughnut',
        data:{labels:[names.revenue.join(' / '),names.cogs.join(' / '),names.gross_profit.join(' / ')],
              datasets:[{data:[raw.revenue,raw.cogs,raw.gross_profit], backgroundColor:['#111','#6b7280','#10b981']}]},
        options:{plugins:{title:{display:true,text:'Composición'}}}
      });
    }
  })();

  // 3) Radar de ratios
  (() => {
    const rkeys = ['gross_margin','operating_margin','net_margin','current_ratio','quick_ratio',
                   'debt_ratio','debt_to_equity','ROA','ROE','asset_turnover'];
    const flat = kpisFlat(k);
    const keys = rkeys.filter(key => flat[key] != null);
    if(!keys.length) return;
    const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
    new Chart(cv, {
      type:'radar',
      data:{
        labels: keys.map(key => names[key].join(' / ')),
        datasets:[{
          label:'Ratios',
          data: keys.map(key => flat[key]),
          backgroundColor:'rgba(17,17,17,.15)',
          borderColor:'#111'
        }]
      },
      options:{scales:{r:{beginAtZero:true}}}
    });
  })();

  // 4) Línea por año
  (() => {
    const years = Object.keys(byYear).sort();
    if(years.length<2) return;
    const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
    const series = ['revenue','operating_income','net_income'].filter(k=>years.some(y=>byYear[y]?.raw?.[k]!=null));
    const datasets = series.map((k,i)=>({
      label:names[k].join(' / '),
      data: years.map(y=>byYear[y]?.raw?.[k]||0),
      borderColor:['#111','#0ea5e9','#10b981'][i%3],
      tension:.2
    }));
    new Chart(cv,{type:'line',
      data:{labels:years, datasets},
      options:{plugins:{title:{display:true,text:'Evolución anual (extraída del PDF)'}}}
    });
  })();

  // 5) Gauges: Liquidez, Apalancamiento, Cobertura intereses
  (() => {
    const flat = kpisFlat(k);
    function gauge(value, target, title){
      if(value==null || isNaN(value)) return;
      const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
      const v = Math.max(0, Math.min(value, target));
      new Chart(cv, {
        type:'doughnut',
        data:{ labels:['Valor','Resto'],
          datasets:[{ data:[v, Math.max(0,target-v)], backgroundColor:['#10b981','#e5e7eb']}]
        },
        options:{ plugins:{title:{display:true, text:title}, legend:{display:false}},
                  cutout:'70%', circumference:180, rotation:270 }
      });
    }
    gauge(flat.current_ratio, 1.5, 'Liquidez (meta 1.5x)');
    if(flat.debt_to_equity!=null){
      const cv = document.createElement('canvas'); chartsBox.appendChild(cv);
      const target = 2.0, val = Math.min(flat.debt_to_equity, target);
      new Chart(cv, {
        type:'doughnut',
        data:{labels:['Dentro meta','Sobre meta'],
          datasets:[{data:[Math.max(0,target-val), val], backgroundColor:['#10b981','#ef4444']}]},
        options:{plugins:{title:{display:true,text:'Apalancamiento (meta ≤ 2,0)'}, legend:{display:false}},
                 cutout:'70%', circumference:180, rotation:270}
      });
    }
    gauge(flat.interest_coverage, 3.0, 'Cobertura intereses (meta 3x)');
  })();


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

    # --- Cupón (usa la tabla COUPONS) ---
    price = float(settings.MP_PRICE_1DAY)         # precio base
    c = COUPONS.get(coupon) if coupon else None   # busca el cupón en la tabla

    # 1) Acceso GRATIS: no requiere MP
    if c and c.get("type") == "free":
        uid = make_user_id(gmail)
        ensure_dirs(user_dir(uid))
        token = make_token(uid, 24)
        return {"skip": True, "token": token}

    # 2) Para el resto, ahora sí exigimos MP configurado
    _require_mp()

    # 3) Descuento porcentual (si aplica) con decimales según moneda
decimals = _currency_decimals(settings.MP_CURRENCY)
if c and c.get("type") == "percent":
    pct = float(c.get("value", 0))
    price = max(1.0, round(price * (100 - pct) / 100, decimals))
# Asegura tipo float/decimal apropiado
price = float(price)

    # 4) Preferencia MP
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

def _currency_decimals(code: str) -> int:
    # CLP/JPY sin decimales; resto 2 por defecto
    return 0 if str(code).upper() in {"CLP","JPY","KRW"} else 2


@app.get("/mp/return", response_class=HTMLResponse)
async def mp_return(status: str | None = None, payment_id: str | None = None, collection_id: str | None = None):
    _require_mp()
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

    secure = "; Secure" if str(settings.BASE_URL).startswith("https://") else ""
    html = f"""
    <script>
      localStorage.setItem('token', '{token}');
      document.cookie = 'token={token}; Path=/; Max-Age=86400; SameSite=Lax{secure}';
      location.href='/portal';
    </script>
    """
    return HTMLResponse(html)



@app.get("/portal", response_class=HTMLResponse)
async def portal(request: Request):
    token = _get_token_from_request(request)
    if not token:
        return HTMLResponse("<script>location.href='/'</script>", status_code=401)
    try:
        read_token_full(token)  # valida y lanza 401 si expiró
    except HTTPException:
        return HTMLResponse("<script>location.href='/'</script>", status_code=401)

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
        # Nombre seguro (evita rutas tipo ../../algo.pdf)
        name = os.path.basename((f.filename or "archivo.pdf").strip())
        if not name.lower().endswith(".pdf"):
            return {"ok": False, "message": f"{name}: solo se aceptan PDF"}

        # Leemos el archivo (en memoria). Si prefieres stream, se puede ajustar.
        content = await f.read()
        total_size += len(content)

        if len(content) > single_max:
            return {"ok": False, "message": f"{name}: supera {settings.SINGLE_FILE_MAX_MB} MB"}

        try:
            # ←← **AQUÍ estaba la indentación mal**
            with open(base/"docs"/name, "wb") as out:
                out.write(content)
        except Exception as e:
            raise HTTPException(500, f"No pude guardar {name}: {e}")

        saved_names.append(name)

        # Límite total por subida: si se excede, deshacemos lo guardado
    if total_size > total_max:
        for nm in saved_names:
            try:
                (base/"docs"/nm).unlink(missing_ok=True)
            except:
                pass
        return {"ok": False, "message": f"Superaste el total permitido ({settings.MAX_TOTAL_MB} MB por subida)."}

    # Indexación pesada en segundo plano (evita timeouts)
    background_tasks.add_task(index_worker, str(base), saved_names)

    # Respuesta estándar para fetch (JSON)
    resp = {
        "ok": True,
        "saved": saved_names,
        "errors": [],
        "indexing": "in_progress",
        "note": "Estamos procesando tus PDFs en segundo plano."
    }


    # Parachoques: si la petición vino esperando HTML (submit tradicional),
    # redirige al portal para que no se quede viendo el JSON.
    accept = (request.headers.get("accept") or "").lower()
    if "text/html" in accept and "application/json" not in accept:
        return RedirectResponse(url="/portal", status_code=303)

    # Para fetch: devuelve JSON
    return resp


@app.post("/upload-zip")
async def upload_zip(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    uid = require_user(request)
    base = user_dir(uid); ensure_dirs(base)

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Sube un .zip con PDFs")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        tmp.write(await file.read()); tmp.close()

        extracted = []
        with ZipFile(tmp.name, 'r') as z:
            for nm in z.namelist():
                if nm.lower().endswith(".pdf"):
                    dest = base/"docs"/Path(nm).name  # evita ZipSlip
                    with z.open(nm) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    extracted.append(dest.name)

        background_tasks.add_task(index_worker, str(base), extracted)
        return {"ok": True, "saved": extracted, "indexing": "in_progress"}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.post("/ingest/url")
async def ingest_from_urls(request: Request, background_tasks: BackgroundTasks, body: Dict[str, Any]):
    uid = require_user(request)
    base = user_dir(uid); ensure_dirs(base)
    urls = body.get("urls") or []
    if not isinstance(urls, list) or not urls:
        raise HTTPException(400, "Incluye 'urls': [ ... ] con enlaces a PDFs")

    max_mb = settings.SINGLE_FILE_MAX_MB
    max_bytes = max_mb * 1024 * 1024
    saved = []

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as ses:
        for u in urls:
            u = str(u).strip()
            if not u.lower().startswith(("http://", "https://")):
                continue
            fn = (base/"docs"/(Path(u).name or "archivo.pdf")).with_suffix(".pdf")
            try:
                async with ses.get(u, allow_redirects=True) as r:
                    if r.status != 200:
                        continue
                    ctype = r.headers.get("Content-Type","").lower()
                    if "pdf" not in ctype:
                        continue
                    # corta por Content-Length si existe
                    cl = r.headers.get("Content-Length")
                    if cl and int(cl) > max_bytes:
                        continue
                    # stream con tope
                    read = 0
                    with open(fn, "wb") as out:
                        async for chunk in r.content.iter_chunked(64*1024):
                            read += len(chunk)
                            if read > max_bytes:
                                raise HTTPException(413, f"Archivo supera {max_mb} MB")
                            out.write(chunk)
                    saved.append(fn.name)
            except Exception:
                # ignora URL con error; si prefieres, acumula en 'errors'
                continue

    if saved:
        background_tasks.add_task(index_worker, str(base), saved)
    return {"ok": True, "saved": saved, "indexing": "in_progress" if saved else "none"}


@app.get("/status")
def status(request: Request):
    uid = require_user(request)
    base = user_dir(uid)
    idx, meta = load_index(base)
    return {"fragments": idx.ntotal, "docs": len({m['doc_title'] for m in meta})}


@app.get("/me")
def me(request: Request):
    token = _get_token_from_request(request)
    if not token:
        raise HTTPException(401, "Falta token")
    data = read_token_full(token)
    now = int(datetime.now(timezone.utc).timestamp())
    remaining = max(0, data.get("exp", 0) - now)
    return {
        "uid": data.get("sub"),
        "code": data.get("code"),
        "exp": data.get("exp"),
        "remaining_seconds": remaining
    }


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

def _pretty_pct(x):
    return f"{x*100:.1f}%" if isinstance(x,(int,float)) else "s/d"

def _band(value, low=None, high=None, reverse=False):
    """
    Clasifica simple: bajo/medio/alto según umbrales.
    reverse=True cuando "más bajo es mejor" (ej: endeudamiento).
    Devuelve ('bueno'|'aceptable'|'riesgo', comentario)
    """
    if value is None:
        return "s/d", "Sin dato."
    if low is None and high is None:
        return "s/d", "Sin umbrales de referencia."
    if reverse:
        # bajo=bueno
        if high is not None and value > high: return "riesgo", f"Alto ({value:.2f})"
        if low is not None and value > low:  return "aceptable", f"Medio ({value:.2f})"
        return "bueno", f"Bajo ({value:.2f})"
    else:
        # alto=bueno
        if low is not None and value < low:  return "riesgo", f"Bajo ({value:.2f})"
        if high is not None and value < high: return "aceptable", f"Medio ({value:.2f})"
        return "bueno", f"Alto ({value:.2f})"

def rule_based_advice_from_kpis(k: dict) -> str:
    """Diagnóstico + recomendaciones a partir de KPIs extraídos del PDF."""
    if not k:
        return ""
    out = []
    out.append("### Definiciones clave")
    out += [
        "- **Liquidez (Razón Corriente)**: capacidad de cubrir pasivos de corto plazo con activos corrientes.",
        "- **Prueba Ácida**: liquidez descontando inventarios ((Activos corrientes − Inventario) / Pasivos corrientes).",
        "- **Margen Bruto/Operacional/Neto**: utilidad como % de ventas en cada nivel.",
        "- **Endeudamiento (Deuda/Activos; Deuda/Patrimonio)**: apalancamiento financiero.",
        "- **Cobertura de Intereses (EBIT/Intereses)**: cuántas veces cubres los intereses.",
        "- **Rotación de Activos (Ventas/Activos)**: eficiencia comercial vs. base de activos.",
    ]

    cr  = k.get("current_ratio")
    qa  = k.get("quick_ratio")
    gm  = k.get("gross_margin")
    om  = k.get("operating_margin")
    nm  = k.get("net_margin")
    dr  = k.get("debt_ratio")
    dte = k.get("debt_to_equity")
    ic  = k.get("interest_coverage")
    at  = k.get("asset_turnover")

    # Umbrales orientativos
    diag = []
    band_cr, cmt_cr = _band(cr, low=1.0, high=1.5, reverse=False); diag.append(f"- **Razón Corriente**: {cr if cr is not None else 's/d'} → {band_cr} ({cmt_cr})")
    band_qa, cmt_qa = _band(qa, low=0.8, high=1.0, reverse=False); diag.append(f"- **Prueba Ácida**: {qa if qa is not None else 's/d'} → {band_qa} ({cmt_qa})")
    band_gm, cmt_gm = _band(gm, low=0.20, high=0.35, reverse=False); diag.append(f"- **Margen Bruto**: {_pretty_pct(gm)} → {band_gm} ({cmt_gm})")
    band_om, cmt_om = _band(om, low=0.05, high=0.15, reverse=False); diag.append(f"- **Margen Operacional**: {_pretty_pct(om)} → {band_om} ({cmt_om})")
    band_nm, cmt_nm = _band(nm, low=0.03, high=0.10, reverse=False); diag.append(f"- **Margen Neto**: {_pretty_pct(nm)} → {band_nm} ({cmt_nm})")
    band_dr, cmt_dr = _band(dr, low=0.50, high=0.70, reverse=True);  diag.append(f"- **Deuda/Activos**: {_pretty_pct(dr) if isinstance(dr,float) else dr} → {band_dr} ({cmt_dr})")
    band_de, cmt_de = _band(dte, low=1.5,  high=2.5,  reverse=True);  diag.append(f"- **Deuda/Patrimonio**: {dte if dte is not None else 's/d'} → {band_de} ({cmt_de})")
    band_ic, cmt_ic = _band(ic, low=1.5,  high=3.0,  reverse=False); diag.append(f"- **Cobertura de Intereses**: {ic if ic is not None else 's/d'} → {band_ic} ({cmt_ic})")
    band_at, cmt_at = _band(at, low=0.6,  high=1.0,  reverse=False); diag.append(f"- **Rotación de Activos**: {at if at is not None else 's/d'} → {band_at} ({cmt_at})")

    out.append("\n### Diagnóstico")
    out += diag

    rec = []
    if band_cr in ("riesgo","aceptable") or band_qa in ("riesgo","aceptable"):
        rec += [
            "- **Liquidez**: acelerar cobros (descuentos por pronto pago), revisar crédito, reducir inventario lento, negociar plazos con proveedores.",
            "- Pausar CAPEX/opex no críticos hasta normalizar capital de trabajo."
        ]
    if band_de in ("riesgo","aceptable") or band_dr in ("riesgo","aceptable"):
        rec += [
            "- **Apalancamiento**: amortizar deuda cara, alargar plazos a menor tasa, evaluar capitalización de utilidades.",
        ]
    if band_om in ("riesgo","aceptable") or band_nm in ("riesgo","aceptable"):
        rec += [
            "- **Margen**: ajustes de precios selectivos, optimizar mix, compras/mermas, eficiencia operativa.",
        ]
    if band_ic in ("riesgo","aceptable"):
        rec += [
            "- **Cobertura de intereses**: renegociar deuda y elevar EBIT con foco en contribución marginal."
        ]
    if not rec:
        rec = ["- Mantener disciplina de costos y rotación; monitorear mensualmente KPIs clave."]

    out.append("\n### Recomendaciones (prioridad)")
    for i, r in enumerate(rec, 1):
        out.append(f"- [P{i}] {r}")

    out.append("\n### Riesgos / alertas")
    out += [
        "- Concentración de ventas/clientes, shocks de tasa/FX, dependencia de insumos clave.",
        "- Calidad de datos: algunos KPIs no estaban disponibles en los PDFs."
    ]

    return "\n".join(out)


def rule_based_advice_from_ctx(question: str, ctx: list[dict]) -> str:
    """
    Usa pasajes (ctx) como evidencia y entrega explicación/diagnóstico simple,
    con mini-resumen y notas por año si se detectan.
    """
    if not ctx:
        return "No hay suficiente evidencia en tus PDFs para responder esa pregunta."

    import re as _reY
    years = []
    for c in ctx:
        y = _reY.findall(r"\b(20\d{2})\b", (c.get("text") or ""))
        years.extend(y)
    years = sorted({y for y in years})

    # Evidencia top-4
    bullets = []
    for c in ctx[:4]:
        doc = c.get("doc_title","?"); pg = c.get("page","?")
        tx  = (c.get("text","") or "").strip().replace("\n"," ")
        if len(tx) > 300: tx = tx[:300] + "…"
        bullets.append(f"- [{doc} p.{pg}] {tx}")

    # Mini-resumen heurístico reutilizando summarize_answer
    try:
        res_like = [(c.get("score",0.0), c) for c in ctx]
        resumen = summarize_answer(question, res_like)
    except Exception:
        resumen = ""

    md = []
    md.append("### Definiciones clave")
    md.append("- *Liquidez*: capacidad para cumplir obligaciones de corto plazo.")
    md.append("- *Endeudamiento*: proporción de deuda sobre activos/patrimonio.")
    md.append("- *Márgenes*: utilidad como % de ventas (bruto/operacional/neto).")
    md.append("- *Cobertura de intereses*: EBIT dividido por gastos financieros.")

    if resumen:
        md.append("\n" + resumen)

    md.append("\n### Diagnóstico (basado en evidencia)")
    md.append("Los pasajes más relevantes para tu pregunta son:")
    md += bullets

    if years:
        md.append("\n### Notas por año (detectado en texto)")
        for y in years[:6]:
            hits = []
            for c in ctx[:10]:
                t = (c.get("text","") or "")
                if str(y) in t:
                    frag = " ".join(t.split())[:220] + "…"
                    hits.append(f"  - **{y}**: {frag}")
                    if len(hits)>=2: break
            md += hits

    md.append("\n### Recomendaciones (prioridad)")
    md.append("- [P1] Capital de trabajo: acelerar cobros y optimizar inventarios según evidencia.")
    md.append("- [P2] Si el apalancamiento es alto, renegocia tasas/plazos y prioriza deuda cara.")
    md.append("- [P3] Margen: precios/mix/mermas y eficiencia operativa.")

    md.append("\n### Riesgos / alertas")
    md.append("- Concentración de clientes, variación de tasas/FX y dependencia de insumos.")

    return "\n".join(md)


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
            # Motor local que explica + diagnostica + recomienda
            answer = rule_based_advice_from_ctx(query, ctx)



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

NAMES = {
    "revenue": ("Ingresos","Revenue"),
    "cogs": ("Costo de ventas","COGS"),
    "gross_profit": ("Utilidad bruta","Gross Profit"),
    "operating_income": ("Resultado operacional (EBIT)","Operating Income"),
    "net_income": ("Utilidad neta","Net Income"),
    "total_assets": ("Activos totales","Total Assets"),
    "total_equity": ("Patrimonio","Equity"),
    "total_liabilities": ("Pasivos totales","Total Liabilities"),
    "current_assets": ("Activos corrientes","Current Assets"),
    "current_liabilities": ("Pasivos corrientes","Current Liabilities"),
    "inventory": ("Inventario","Inventory"),
    "accounts_receivable": ("Cuentas por cobrar","Accounts Receivable"),
    "accounts_payable": ("Cuentas por pagar","Accounts Payable"),
    "interest_expense": ("Gastos financieros","Interest Expense"),
    "ebitda": ("EBITDA","EBITDA"),
}

GLOSARIO = {
    "EBIT": "Earnings Before Interest and Taxes (Resultado de explotación).",
    "EBITDA": "EBIT + Depreciación y Amortización.",
    "COGS": "Cost of Goods Sold (Costo de ventas).",
    "AV/AH": "Análisis Vertical / Horizontal.",
    "CFO/CFI/CFF": "Flujos de Caja de Operación / Inversión / Financiamiento.",
    "ROA": "Return on Assets (Utilidad/Activos).",
    "ROE": "Return on Equity (Utilidad/Patrimonio)."
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
    from collections import defaultdict

    values: Dict[str, float] = {}
    values_by_year: Dict[str, Dict[str, float]] = defaultdict(dict)
    YR = _re.compile(r"\b(20\d{2})\b")  # ojo: raw string correcto

    for pdf in pdfs:
        try:
            with pdfplumber.open(pdf) as doc:
                for page in doc.pages:
                    text = page.extract_text() or ""
                    low = text.lower()
                    for key, keys in LABELS.items():
                        for kw in keys:
                            if kw in low:
                                for ln in text.splitlines():
                                    if kw in ln.lower():
                                        v = _num(ln)
                                        if v is None:
                                            continue
                                        # guarda primer valor visto
                                        values.setdefault(key, v)
                                        # intenta capturar año en la misma línea
                                        m = YR.search(ln)
                                        if m:
                                            yr = m.group(1)
                                            values_by_year[yr].setdefault(key, v)
        except Exception:
            continue

    out: Dict[str, Any] = {"raw": values}

    # --------- Conveniencias (derivados globales) ----------
    rev    = values.get("revenue")
    cogs   = values.get("cogs")
    gp     = values.get("gross_profit")
    op     = values.get("operating_income")
    ni     = values.get("net_income")
    assets = values.get("total_assets")
    eq     = values.get("total_equity")
    liab   = values.get("total_liabilities")
    ca     = values.get("current_assets")
    cl     = values.get("current_liabilities")
    inv    = values.get("inventory")
    ar     = values.get("accounts_receivable")
    ap     = values.get("accounts_payable")
    intexp = values.get("interest_expense")
    ebitda = values.get("ebitda")

    # Márgenes
    if rev and cogs: out["gross_margin"] = (rev - cogs) / rev
    if op and rev:   out["operating_margin"] = op / rev
    if ni and rev:   out["net_margin"] = ni / rev

    # Liquidez
    if ca and cl and cl != 0: out["current_ratio"] = ca / cl
    if ca is not None and inv is not None and cl and cl != 0:
        out["quick_ratio"] = (ca - inv) / cl

    # Endeudamiento
    if liab and assets and assets != 0: out["debt_ratio"] = liab / assets
    if liab and eq and eq != 0:         out["debt_to_equity"] = liab / eq
    if op and intexp and intexp != 0:   out["interest_coverage"] = op / intexp

    # Rentabilidad
    if ni and assets and assets != 0: out["ROA"] = ni / assets
    if ni and eq and eq != 0:         out["ROE"] = ni / eq
    if rev and assets and assets != 0: out["asset_turnover"] = rev / assets

    # Actividad (días)
    if rev and ar:   out["days_receivable"] = 365 * ar / rev
    if cogs and ap:  out["days_payable"] = 365 * ap / cogs
    if cogs and inv: out["inventory_turnover"] = cogs / max(inv, 1e-9)

    # Working capital
    if ca is not None and cl is not None:
        out["working_capital"] = ca - cl

    # --------- Derivados por año ----------
    by_year: Dict[str, Any] = {}
    for yr, vv in values_by_year.items():
        yy = {"raw": vv}
        rev    = vv.get("revenue");         cogs   = vv.get("cogs")
        op     = vv.get("operating_income"); ni    = vv.get("net_income")
        assets = vv.get("total_assets");     eq    = vv.get("total_equity")
        liab   = vv.get("total_liabilities"); ca   = vv.get("current_assets")
        cl     = vv.get("current_liabilities"); inv = vv.get("inventory")

        if rev and cogs: yy["gross_margin"] = (rev - cogs) / rev
        if op and rev:   yy["operating_margin"] = op / rev
        if ni and rev:   yy["net_margin"] = ni / rev

        if ca and cl and cl != 0: yy["current_ratio"] = ca / cl
        if ca is not None and inv is not None and cl and cl != 0:
            yy["quick_ratio"] = (ca - inv) / cl

        if liab and assets and assets != 0: yy["debt_ratio"] = liab / assets
        if liab and eq and eq != 0:         yy["debt_to_equity"] = liab / eq

        if ni and assets and assets != 0: yy["ROA"] = ni / assets
        if ni and eq and eq != 0:         yy["ROE"] = ni / eq
        if rev and assets and assets != 0: yy["asset_turnover"] = rev / assets

        by_year[yr] = yy

    out["by_year"] = by_year
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
    for key in ["revenue","cogs","gross_profit","operating_income","net_income",
                "total_assets","total_equity","total_liabilities","current_assets",
                "current_liabilities","inventory","accounts_receivable",
                "accounts_payable","ebitda","interest_expense","working_capital"]:
        if key in k.get("raw", {}):
            es, en = NAMES.get(key, (key, key))
            lines.append(f"- {es} / {en}: {val(k['raw'][key])}")

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

    if k.get("by_year"):
        lines.append("")
        lines.append("## 6.1) KPIs por año (extraídos del PDF)")
        years = sorted(k["by_year"].keys())
        for y in years:
            yy = k["by_year"][y]
            lines.append(
                f"- **{y}**: " + ", ".join(
                    f"{NAMES.get(m,(m,m))[0]}: {val((yy.get('raw') or {}).get(m))}"
                    for m in ["revenue","operating_income","net_income"]
                    if (yy.get('raw') or {}).get(m) is not None
                )
            )

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

    # --- 12.1) ¿Qué significa cada KPI? ---
    EXPLAIN = {
        "current_ratio": ("Razón Corriente", "Cuántas veces los activos de corto plazo alcanzan para pagar las deudas de corto plazo. Ideal ≥ 1,5."),
        "quick_ratio": ("Prueba Ácida", "Igual que la razón corriente pero sin contar inventarios (más exigente). Ideal ≥ 1,0."),
        "gross_margin": ("Margen Bruto", "Por cada $1 de ventas, cuánto queda tras el costo de los productos vendidos."),
        "operating_margin": ("Margen Operacional", "Utilidad de la operación por cada $1 de ventas (sin considerar impuestos/financieros)."),
        "net_margin": ("Margen Neto", "Utilidad final por cada $1 de ventas (después de todo)."),
        "debt_ratio": ("Deuda/Activos", "Qué parte de los activos está financiada con deuda. Más bajo = mejor colchón."),
        "debt_to_equity": ("Deuda/Patrimonio", "Cuántos pesos de deuda hay por cada peso de patrimonio. Más bajo reduce riesgo."),
        "interest_coverage": ("Cobertura de Intereses", "Cuántas veces el EBIT cubre los intereses. Ideal ≥ 3."),
        "ROA": ("ROA", "Rentabilidad de los activos: utilidad que generan los activos invertidos."),
        "ROE": ("ROE", "Rentabilidad del patrimonio: utilidad para los dueños respecto a su inversión."),
        "asset_turnover": ("Rotación de Activos", "Eficiencia comercial respecto al tamaño de los activos."),
        "days_receivable": ("Días de Cobro", "En promedio, cuántos días demoras en cobrar a clientes."),
        "days_payable": ("Días de Pago", "En promedio, cuántos días demoras en pagar a proveedores."),
        "inventory_turnover": ("Rotación de Inventario", "Veces en el año que renuevas el inventario.")
    }
    lines.append("")
    lines.append("## 12.1) ¿Qué significa cada KPI?")
    for key, (title, note) in EXPLAIN.items():
        val_show = k.get(key)
        if key in ("gross_margin","operating_margin","net_margin","ROA","ROE","debt_ratio"):
            val_fmt = pct(val_show)
        else:
            val_fmt = val_show if val_show is not None else "s/d"
        lines.append(f"- **{title}**: {note} — *Valor detectado:* {val_fmt}")

    lines.append("")
    lines.append("## Glosario")
    for kgl, vgl in GLOSARIO.items():
        lines.append(f"- **{kgl}**: {vgl}")

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
    executive_summary = premium_exec_summary(period, k)  # None si no hay OPENAI_API_KEY

    result_md = "\n".join(lines)
    if executive_summary:
        result_md = "## Resumen Ejecutivo (IA)\n" + executive_summary + "\n\n" + result_md

    # Añade diagnóstico local (definiciones + evaluación + recomendaciones)
    diag_md = rule_based_advice_from_kpis(k)
    if diag_md:
        result_md += "\n\n---\n\n" + diag_md

    return {
        "kpis": k,
        "report_markdown": result_md,
        "executive_summary": executive_summary
    }



# Salud

@app.get("/__check_coupon")
def __check_coupon():
    return {
        "coupon_field_in_template": 'name="coupon"' in BASE_HTML
    }


@app.get("/health")
async def health():
    if SETTINGS_ERROR:
        return PlainTextResponse(f"config error: {SETTINGS_ERROR}", status_code=500)
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
