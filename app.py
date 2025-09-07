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

import mercadopago
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
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

    STORAGE_DIR: str = "storage"
    MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 200
    MIN_CHARS: int = 200
    TOP_K: int = 6
    NORMALIZE: bool = True

    class Config:
        env_file = ".env"

# Cargar settings con manejo de error para que /health lo muestre si falla
SETTINGS_ERROR = None
try:
    settings = Settings()
except Exception as e:
    SETTINGS_ERROR = str(e)
    settings = None

app = FastAPI(title=(settings.APP_NAME if settings else "Finance"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Mercado Pago SDK
mp = mercadopago.SDK(settings.MP_ACCESS_TOKEN)
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
        _model = SentenceTransformer(settings.MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = get_model().encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=settings.NORMALIZE)
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
    p = idx_paths(base)
    dim = get_model().get_sentence_embedding_dimension()
    if p["faiss"].exists() and p["meta"].exists():
        idx = faiss.read_index(str(p["faiss"]))
        meta = [json.loads(x) for x in open(p["meta"], "r", encoding="utf-8") if x.strip()]
        return idx, meta
    return faiss.IndexFlatIP(dim), []

def save_index(base: Path, idx, meta: List[Dict[str, Any]]):
    p = idx_paths(base)
    p["faiss"].parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(p["faiss"]))
    with open(p["meta"], "w", encoding="utf-8") as f:
        for m in meta: f.write(json.dumps(m, ensure_ascii=False) + "\n")

def add_pdfs_to_index(base: Path, pdfs: List[Path]) -> int:
    ensure_dirs(base)
    idx, meta = load_index(base)
    new_txt, new_meta = [], []
    for pdf in pdfs:
        if not pdf.exists(): continue
        with fitz.open(pdf) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = clean_text(page.get_text("text") or "")
                if len(text) < 50: continue
                for ck in chunk_text(text):
                    new_txt.append(ck)
                    new_meta.append({"doc_title": pdf.name, "page": page_num, "text": ck})
    if not new_txt: return 0
    vecs = embed_texts(new_txt)
    idx.add(vecs); meta.extend(new_meta)
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

# ===================== HTML =====================
BASE_HTML = f"""
<!doctype html><html lang=es><head>
<meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{settings.APP_NAME}</title>
<style>body{{font-family:system-ui;margin:2rem}} .card{{max-width:860px;margin:auto;padding:1.2rem 1.5rem;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.06)}} input,button{{padding:.6rem .8rem;border-radius:10px;border:1px solid #d1d5db;width:100%}} button{{background:#111;color:#fff;border:none;cursor:pointer}} .row{{display:flex;gap:12px;flex-wrap:wrap}} .muted{{color:#6b7280}}</style>
<script>
async function startCheckout(ev){{
  ev.preventDefault();
  const f = ev.target.closest('form');
  const nombre = f.nombre.value.trim();
  const apellido = f.apellido.value.trim();
  const gmail = f.gmail.value.trim();
  if(!/^[^@\s]+@gmail\.com$/.test(gmail)){{alert('Ingresa un Gmail válido');return;}}
  const r = await fetch('/mp/create-preference',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{nombre,apellido,gmail}})}});
  const data = await r.json();
  if(!data.init_point){{alert('No se pudo crear la preferencia');return;}}
  location.href = data.init_point; // redirige a Checkout Pro
}}
</script>
</head><body>
<div class=card>
<h1>{settings.APP_NAME}</h1>
<p class=muted>Acceso por 24h tras el pago con Mercado Pago. Sube tus informes PDF y obtén KPI + análisis + sugerencias. Para asesoría completa: <b>germanperey@gmail.com</b>.</p>
<form onsubmit="startCheckout(event)">
  <div class=row>
    <input name=nombre placeholder="Nombre" required>
    <input name=apellido placeholder="Apellido" required>
    <input name=gmail placeholder="Gmail (obligatorio)" required>
  </div>
  <div style="margin-top:12px"><button>Pagar y acceder por 1 día</button></div>
</form>
</div>
</body></html>
"""

PORTAL_HTML = """
<!doctype html><html lang=es><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Portal</title>
<style>body{font-family:system-ui;margin:2rem} .card{max-width:1000px;margin:auto;padding:1.2rem 1.5rem;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.06)} input,button,textarea{padding:.6rem .8rem;border-radius:10px;border:1px solid #d1d5db;width:100%} button{background:#111;color:#fff;border:none;cursor:pointer} pre{white-space:pre-wrap;background:#f9fafb;padding:1rem;border-radius:10px;border:1px solid #eee} .row{display:flex;gap:12px;flex-wrap:wrap}</style>
</head><body>
<div class=card>
<h2>Portal de usuario (pase activo)</h2>
<form id=up method=post enctype=multipart/form-data>
  <input type=file name=files multiple accept=application/pdf required>
  <div style="margin-top:8px"><button>Subir PDFs e indexar</button></div>
</form>
<pre id=upres></pre>
<hr>
<form id=askf>
  <input name=q placeholder="Tu pregunta" required>
  <div style="margin-top:8px"><button>Preguntar</button></div>
</form>
<pre id=askres></pre>
<hr>
<form id=rep>
  <input name=periodo placeholder="Periodo (opcional)">
  <div style="margin-top:8px"><button>Generar Reporte Automático</button></div>
</form>
<pre id=repres></pre>
</div>
<script>
async function withToken(url,opts={{}}){const t=localStorage.getItem('token');opts.headers=Object.assign({'Authorization':'Bearer '+t},opts.headers||{});return fetch(url,opts)}
up.onsubmit=async e=>{e.preventDefault();const fd=new FormData(up);const r=await withToken('/upload',{method:'POST',body:fd});upres.textContent=await r.text()}
askf.onsubmit=async e=>{e.preventDefault();const q=new FormData(askf).get('q');const r=await withToken('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,top_k:6})});askres.textContent=await r.text()}
rep.onsubmit=async e=>{e.preventDefault();const periodo=new FormData(rep).get('periodo')||'';const r=await withToken('/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({period:periodo})});repres.textContent=await r.text()}
</script>
</body></html>
"""

# ===================== Rutas públicas =====================
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(BASE_HTML)

@app.post("/mp/create-preference")
async def mp_create_preference(payload: Dict[str, str]):
    nombre = payload.get("nombre", "").strip()
    apellido = payload.get("apellido", "").strip()
    gmail = payload.get("gmail", "").strip().lower()
    if not nombre or not apellido:
        raise HTTPException(400, "Nombre y Apellido son obligatorios")
    if not re.match(r"^[^@\s]+@gmail\.com$", gmail):
        raise HTTPException(400, "Debes usar un correo @gmail.com válido")

    preference = {
        "items": [{
            "title": f"Pase 1 día — {settings.APP_NAME}",
            "quantity": 1,
            "currency_id": settings.MP_CURRENCY,
            "unit_price": float(settings.MP_PRICE_1DAY),
        }],
        "payer": {"email": gmail},                   # IMPORTANTE
        "back_urls": {
            "success": f"{settings.BASE_URL}/mp/return",
            "failure": f"{settings.BASE_URL}/mp/return",
            "pending":  f"{settings.BASE_URL}/mp/return",
        },
        "auto_return": "approved",
        "purpose": "wallet_purchase",
        "metadata": {"gmail": gmail, "nombre": nombre, "apellido": apellido},
        "external_reference": hashlib.sha1(gmail.encode()).hexdigest(),
    }
    try:
        pref = mp.preference().create(preference)["response"]
        return {
            "id": pref.get("id"),
            "init_point": pref.get("init_point") or pref.get("sandbox_init_point"),
        }
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
    return HTMLResponse(PORTAL_HTML)

# (Opcional) Webhook para notificaciones asincrónicas
# @app.post("/mp/webhook")
# async def mp_webhook(request: Request):
#     body = await request.json()
#     # procesa body['data']['id'] cuando type = payment
#     return {"received": True}

# ===================== Rutas protegidas =====================
from fastapi import Depends

def require_user(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "Falta token")
    token = auth.split(" ",1)[1]
    return read_token(token)

@app.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    uid = require_user(request)
    base = user_dir(uid); ensure_dirs(base)
    saved: List[Path] = []
    for f in files:
        if not f.filename.lower().endswith('.pdf'):
            raise HTTPException(400, 'Solo PDFs')
        dest = base/"docs"/f.filename
        content = await f.read()
        with open(dest, 'wb') as out: out.write(content)
        saved.append(dest)
    fragments = add_pdfs_to_index(base, saved)
    return {"uploaded": [p.name for p in saved], "fragments_indexed": fragments}

@app.post("/ask")
async def ask(request: Request, body: Dict[str, Any]):
    uid = require_user(request)
    q = body.get("question", "")
    if not q: raise HTTPException(400, "Falta 'question'")
    res = semantic_search(user_dir(uid), q, settings.TOP_K)
    if not res:
        return {"answer": None, "context": [], "message": "Sin documentos indexados"}
    ctx = [{"score": round(s,3), **m} for s,m in res]
    answer = "\n\n".join([f"[Fuente: {c['doc_title']} p.{c['page']}]\n{c['text']}" for c in ctx])
    return {"answer": "Pasajes más relevantes:", "context": ctx, "evidence": answer}

# ---- KPI extractor (igual que antes, simplificado) ----
import re as _re
LABELS = {
    "revenue": ["ingresos","ventas","ventas netas"],
    "cogs": ["costo de ventas"],
    "gross_profit": ["utilidad bruta"],
    "operating_income": ["resultado operacional","utilidad de operación"],
    "net_income": ["utilidad neta","resultado del ejercicio"],
    "total_assets": ["activos totales"],
    "total_equity": ["patrimonio"],
    "total_liabilities": ["pasivos totales"],
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
    rev = values.get("revenue"); cogs = values.get("cogs")
    gp = values.get("gross_profit"); op = values.get("operating_income")
    ni = values.get("net_income"); assets = values.get("total_assets"); eq = values.get("total_equity")
    if rev and cogs: out["gross_margin"]=(rev-cogs)/rev
    if op and rev: out["operating_margin"]=op/rev
    if ni and rev: out["net_margin"]=ni/rev
    if ni and assets: out["ROA"]=ni/assets
    if ni and eq and eq!=0: out["ROE"]=ni/eq
    return out

@app.post("/report")
async def report(request: Request, body: Dict[str, Any]):
    uid = require_user(request)
    base = user_dir(uid)
    pdfs = list((base/"docs").glob("*.pdf"))
    if not pdfs: return {"message":"Sube al menos un PDF"}
    kpis = extract_kpis_from_pdfs(pdfs)
    analysis = []
    if "gross_margin" in kpis: analysis.append(f"Margen bruto ~ {kpis['gross_margin']*100:.1f}%")
    if "operating_margin" in kpis: analysis.append(f"Margen operacional ~ {kpis['operating_margin']*100:.1f}%")
    if "net_margin" in kpis: analysis.append(f"Margen neto ~ {kpis['net_margin']*100:.1f}%")
    if "ROE" in kpis: analysis.append(f"ROE ~ {kpis['ROE']*100:.1f}%")
    if "ROA" in kpis: analysis.append(f"ROA ~ {kpis['ROA']*100:.1f}%")
    suggestions = [
        "Optimiza capital de trabajo (cobranza, inventario, pagos).",
        "Revisa estructura de costos para mejorar margen operacional.",
        "Equilibra apalancamiento y riesgo (cobertura de intereses).",
        "Planifica caja con escenarios y colchón de liquidez.",
        "Define OKRs (margen, ROE) y monitorea mensual.",
    ]
    return {"kpis": kpis, "analysis": analysis, "suggestions": suggestions, "contact": "germanperey@gmail.com"}

# Salud
from fastapi.responses import HTMLResponse, PlainTextResponse

@app.get("/health")
async def health():
    if SETTINGS_ERROR:
        # Devuelve el error de arranque para diagnosticar (temporal)
        return PlainTextResponse("settings_error: " + SETTINGS_ERROR, status_code=500)
    return PlainTextResponse("ok", status_code=200)
