#!/usr/bin/env python3
"""
Asesor Financiero – Portal Seguro y Reporte Interactivo

Esta aplicación expone una interfaz web y un backend para que los
usuarios suban sus archivos PDF de análisis económico–financiero, los
indexen de forma incremental y hagan consultas mediante preguntas en
lenguaje natural. También genera un reporte automático con gráficos
e indicadores clave de desempeño (KPI) bilingües, e incluye un
glosario para abreviaturas financieras. La ruta del portal está
protegida para que solo accedan usuarios autenticados mediante un
token JWT recibido tras el pago o cupón. Las cargas de PDFs no
eliminan los archivos previamente subidos y se procesan en segundo
plano para evitar tiempos de espera largos.

Principales características:

* Autenticación con JWT (bearer o cookie) y protección del portal
  (/portal) ante accesos no autorizados.
* Carga de archivos PDF de a uno o varios; las subidas sucesivas
  agregan documentos al índice sin borrar los existentes.
* Mensajes de retroalimentación amigables: tras subir se indica
  claramente qué archivos se están procesando.
* Motor de RAG (retrieval‑augmented generation) por usuario con
  extracción de fragmentos y metadatos, incluyendo detección de
  años en el texto para poder filtrar respuestas por periodo.
* Endpoint /ask para contestar hasta cinco preguntas de manera
  independiente, filtrando por años cuando se solicitan intervalos
  (p. ej. “2020–2023”) y evitando respuestas genéricas si no se
  encuentra evidencia suficiente.
* Endpoint /auto-report que devuelve datos tabulares y definiciones
  para construir múltiples gráficos (barras, líneas y pie) y
  tarjetas KPI con semáforo, fórmulas y acciones sugeridas.
* Glosario de abreviaturas comúnmente usadas en finanzas.

Al integrar este archivo en tu proyecto y actualizar el frontend
embebido, tendrás un portal más robusto, seguro y amigable para los
usuarios finales.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import mercadopago
from fastapi import (BackgroundTasks, Depends, FastAPI, File, HTTPException,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jose import JWTError, jwt
from pydantic_settings import BaseSettings

# ===================== Configuración =====================

class Settings(BaseSettings):
    """Parámetros de configuración cargados desde variables de entorno."""
    # Mercado Pago / cobranzas
    MP_ACCESS_TOKEN: str
    MP_PUBLIC_KEY: str
    MP_PRICE_1DAY: float
    MP_CURRENCY: str = "CLP"

    # Claves y parámetros generales
    JWT_SECRET: str
    BASE_URL: str = "http://localhost:8000"
    APP_NAME: str = "Asesor Financiero"

    # OpenAI (opcional, no usado en local)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Almacenamiento e índice
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


# Intenta cargar configuración al inicio; en caso de error se guarda
# para devolver en /health si es necesario.
SETTINGS_ERROR = None
try:
    settings = Settings()
except Exception as e:
    SETTINGS_ERROR = str(e)
    settings = None

APP_NAME_SAFE = (settings.APP_NAME if settings else "Asesor Financiero")

app = FastAPI(title=(settings.APP_NAME if settings else "Finance"))

# Permitir orígenes cruzados; incluir base_url y variantes http/https
allowed = {
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Si tu portal está en un dominio propio agrégalo aquí
}
if settings and getattr(settings, "BASE_URL", None):
    allowed.add(settings.BASE_URL)
    if settings.BASE_URL.startswith("https://"):
        allowed.add(settings.BASE_URL.replace("https://", "http://"))
    if settings.BASE_URL.startswith("http://"):
        allowed.add(settings.BASE_URL.replace("http://", "https://"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(allowed),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================ Utilidades de autenticación =================

def make_token(uid: str, hours: int = 24) -> str:
    """Genera un JWT para el usuario con expiración en horas."""
    exp = datetime.now(timezone.utc) + timedelta(hours=hours)
    payload = {"uid": uid, "exp": exp.timestamp()}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def read_token(token: str) -> str:
    """Valida y decodifica un token JWT, devolviendo el UID."""
    try:
        data = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        uid = data.get("uid")
        if not uid:
            raise ValueError("Token sin uid")
        exp_ts = data.get("exp")
        if exp_ts and datetime.now(timezone.utc).timestamp() > exp_ts:
            raise ValueError("Token expirado")
        return uid
    except JWTError:
        raise HTTPException(401, "Token inválido")


def require_user(request: Request) -> str:
    """Extrae y valida el token del encabezado Authorization o de una cookie."""
    # Primero intenta Authorization: Bearer <token>
    auth = request.headers.get("Authorization", "")
    token: Optional[str] = None
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    # Si no hay header, intenta cookie
    if not token:
        token = request.cookies.get("token")
    if not token:
        raise HTTPException(401, "No autenticado")
    return read_token(token)


# ================ Embeddings y RAG =================

_model = None


def get_model():
    """Carga el modelo de embeddings de forma perezosa."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.MODEL_NAME)
    return _model


def embed_texts(texts: List[str]):
    """Genera embeddings para una lista de textos y los devuelve como array float32."""
    import numpy as np
    vecs = get_model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=settings.NORMALIZE,
    )
    return np.ascontiguousarray(vecs.astype("float32"))


def clean_text(t: str) -> str:
    """Normaliza el texto eliminando caracteres nulos y espacios extra."""
    t = t.replace("\x00", " ")
    t = "\n".join([ln.strip() for ln in t.splitlines()])
    t = " ".join(t.split())
    return t.strip()


def chunk_text(text: str) -> List[str]:
    """Divide un texto largo en fragmentos superpuestos aptos para embeddings."""
    text = text.strip()
    if not text:
        return []
    size, overlap = settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    chunks: List[str] = []
    start = 0
    N = len(text)
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
        if start == end:
            break
    return chunks


def make_user_id(gmail: str) -> str:
    """Genera un identificador determinista basado en el Gmail."""
    return hashlib.sha256(gmail.lower().encode()).hexdigest()[:16]


def user_dir(uid: str) -> Path:
    """Directorio base de un usuario para docs e índices."""
    return Path(settings.STORAGE_DIR) / uid


def ensure_dirs(base: Path):
    """Crea directorios de almacenamiento e índice si no existen."""
    (base / "docs").mkdir(parents=True, exist_ok=True)
    (base / ".rag_index").mkdir(parents=True, exist_ok=True)


# ====== Persistencia e índice FAISS por usuario ======

def idx_paths(base: Path) -> Dict[str, Path]:
    return {
        "faiss": base / ".rag_index" / "index.faiss",
        "meta": base / ".rag_index" / "metadata.jsonl",
    }


def load_index(base: Path):
    """Carga el índice FAISS y la metadata; si no existe, crea uno vacío."""
    import faiss
    p = idx_paths(base)
    dim = get_model().get_sentence_embedding_dimension()
    if p["faiss"].exists() and p["meta"].exists():
        idx = faiss.read_index(str(p["faiss"]))
        meta = [json.loads(x) for x in open(p["meta"], "r", encoding="utf-8") if x.strip()]
        return idx, meta
    # índice plano de producto interno (cosine si normalizas)
    idx = faiss.IndexFlatIP(dim)
    return idx, []


def save_index(base: Path, idx, meta: List[Dict[str, Any]]):
    """Persistencia del índice FAISS y metadata."""
    import faiss
    p = idx_paths(base)
    p["faiss"].parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(p["faiss"]))
    with open(p["meta"], "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def guess_year_from_text(text: str) -> Optional[int]:
    """Intenta extraer el primer año de cuatro dígitos del texto (1900–2099)."""
    for match in re.findall(r"\b(19|20)\d{2}\b", text):
        try:
            year = int(match)
            if 1900 <= year <= 2099:
                return year
        except Exception:
            pass
    return None


def add_pdfs_to_index(base: Path, pdfs: List[Path]) -> int:
    """Abre una lista de PDFs, divide en fragmentos, calcula embeddings y
    añade al índice existente, guardando metadata con doc y año."""
    import fitz  # PyMuPDF
    ensure_dirs(base)
    idx, meta = load_index(base)
    new_txts: List[str] = []
    new_meta: List[Dict[str, Any]] = []
    for pdf in pdfs:
        try:
            doc = fitz.open(pdf)
        except Exception:
            continue
        full_text = ""
        for page in doc:
            full_text += "\n" + (page.get_text() or "")
        doc.close()
        full_text = clean_text(full_text)
        fragments = chunk_text(full_text)
        for frag in fragments:
            yr = guess_year_from_text(frag) or guess_year_from_text(pdf.name) or None
            new_txts.append(frag)
            new_meta.append({"text": frag, "doc": pdf.name, "year": yr})
    if new_txts:
        vecs = embed_texts(new_txts)
        idx.add(vecs)
        meta.extend(new_meta)
        save_index(base, idx, meta)
    return len(new_txts)


def search_faiss(base: Path, query: str, k: int) -> List[Tuple[str, Dict[str, Any], float]]:
    """Busca los fragmentos más relevantes para una consulta en el índice del usuario."""
    import numpy as np
    idx, meta = load_index(base)
    if idx.ntotal == 0 or not meta:
        return []
    qvec = embed_texts([query])
    D, I = idx.search(qvec, k)
    hits: List[Tuple[str, Dict[str, Any], float]] = []
    for score, i in zip(D[0], I[0]):
        if i < len(meta):
            hits.append((meta[i]["text"], meta[i], float(score)))
    return hits


def parse_years(q: str) -> List[int]:
    """Extrae años o rangos de años de una consulta."""
    years: List[int] = []
    # rango "2020-2023" o "2020–2023"
    m = re.search(r"\b(19|20)\d{2}\s*[\-–]\s*(19|20)\d{2}\b", q)
    if m:
        a = int(m.group(0)[:4])
        b = int(m.group(0)[-4:])
        if a > b:
            a, b = b, a
        years = list(range(a, b + 1))
    else:
        years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", q)]
    # elimina duplicados y ordena
    years = sorted({y for y in years if 1900 <= y <= 2099})
    return years


# ================ Plantilla del portal =================

PORTAL_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Portal de usuario (pase activo)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; margin: 1rem; }
    h1 { font-size: 1.4rem; margin-bottom: 0.5rem; }
    .btn { background: #111; color: #fff; padding: 0.5rem 1rem; border: none; cursor: pointer; border-radius: 4px; margin-top: 0.5rem; }
    .btn:hover { background: #333; }
    #pending { margin-top: 0.3rem; padding: 0; list-style: none; }
    #pending li { margin: 0.1rem 0; font-size: 0.9rem; }
    #pending a { color: #b91c1c; margin-left: 0.5rem; text-decoration: none; }
    #pending a:hover { text-decoration: underline; }
    .nowrap { white-space: nowrap; display: inline-block; }
    #report { margin-top: 1rem; }
    #charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem; }
    #kpi-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.5rem; margin-top: 1rem; }
    .kpi { padding: 0.6rem; border-radius: 0.6rem; border: 1px solid #eee; font-size: 0.9rem; }
    .kpi.ok { background: #e6ffed; }
    .kpi.warn { background: #fff8db; }
    .kpi.bad { background: #ffecec; }
    @media print {
      #file-section, #ask-section, #auto-btn, #premium, #pending, #pending-wrapper, #upload-btn, #ask-btn, #auto-btn, #print-btn { display: none !important; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>{APP_NAME}</h1>
  <button id="print-btn" class="btn" onclick="window.print()">Imprimir reporte</button>
  <div id="file-section">
    <label for="filepick" class="btn">Elegir archivos</label>
    <input id="filepick" type="file" accept="application/pdf" multiple style="display:none" />
    <ul id="pending"></ul>
    <button id="upload-btn" class="btn">Subir PDFs e indexar</button>
    <p style="font-size:0.8rem;color:#444">Límites: máx [MAX_FILES] PDFs por subida · [SINGLE_MAX] MB cada uno · hasta [TOTAL_MAX] MB en total.</p>
    <p id="upload-msg" style="font-size:0.85rem;color:#1e40af"></p>
  </div>
  <div id="ask-section" style="margin-top:1rem;">
    <textarea id="questions" rows="5" style="width:100%;" placeholder="Escribe hasta 5 preguntas, una por línea"></textarea>
    <label class="nowrap" id="premium" style="margin-top:0.5rem;"><input type="checkbox" id="prosa" /> Prosa premium (IA)</label>
    <button id="ask-btn" class="btn">Preguntar</button>
    <div id="answers" style="margin-top:1rem;font-size:0.9rem;"></div>
  </div>
  <button id="auto-btn" class="btn" style="margin-top:1rem;">Generar Reporte Automático</button>
  <div id="report"></div>

<script>
// ------- Cliente JS -------
const pending = [];
const fileInput = document.getElementById('filepick');
const pendingEl = document.getElementById('pending');
const uploadBtn = document.getElementById('upload-btn');
const uploadMsg = document.getElementById('upload-msg');
const askBtn = document.getElementById('ask-btn');
const autoBtn = document.getElementById('auto-btn');

fileInput.onchange = () => {
  for (const f of fileInput.files) {
    if (!pending.some(p => p.file.name === f.name)) {
      pending.push({file: f, name: f.name});
    }
  }
  renderPending();
  fileInput.value = '';
};

function renderPending() {
  pendingEl.innerHTML = pending.map(p => `<li>${p.name} <a href="#" onclick="removeFile('${p.name}'); return false;">✖</a></li>`).join('');
}

function removeFile(name) {
  const idx = pending.findIndex(p => p.name === name);
  if (idx >= 0) { pending.splice(idx, 1); renderPending(); }
}

uploadBtn.onclick = async () => {
  if (pending.length === 0) { alert('No hay archivos para subir.'); return; }
  const fd = new FormData();
  for (const p of pending) fd.append('files', p.file);
  uploadBtn.disabled = true;
  try {
    const r = await fetch('/upload', { method:'POST', body: fd, headers: authHeader() });
    const data = await r.json();
    if (data.ok) {
      // Elimina del buffer solo los que se guardaron
      for (const nm of (data.saved || [])) {
        const i = pending.findIndex(p => p.name === nm);
        if (i >= 0) pending.splice(i,1);
      }
      renderPending();
      const lista = (data.saved || []).map(n => `"${n}"`).join(', ');
      uploadMsg.textContent = `Estamos cargando tus archivo(s) PDF: ${lista}. Analizándolos en este instante.`;
    } else {
      uploadMsg.textContent = data.message || 'Error al subir los archivos.';
    }
  } catch (err) {
    uploadMsg.textContent = 'Error de red al subir.';
  } finally {
    uploadBtn.disabled = false;
  }
};

askBtn.onclick = async () => {
  const qs = document.getElementById('questions').value.split('\n').filter(l => l.trim());
  if (qs.length === 0) { alert('Ingresa una o más preguntas.'); return; }
  if (qs.length > 5) { alert('Máximo 5 preguntas.'); return; }
  askBtn.disabled = true;
  const answersEl = document.getElementById('answers');
  answersEl.innerHTML = '';
  try {
    const r = await fetch('/ask', {
      method:'POST',
      headers: { ...authHeader(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ questions: qs })
    });
    const data = await r.json();
    if (!Array.isArray(data)) {
      answersEl.textContent = data.message || 'Error al preguntar.';
    } else {
      answersEl.innerHTML = data.map(obj => {
        const q = obj.question;
        const ans = obj.answer;
        return `<div style='margin-bottom:1rem;'><b>${q}</b><br>${ans.replace(/\n/g,'<br>')}</div>`;
      }).join('');
    }
  } catch (err) {
    answersEl.textContent = 'Error de red al preguntar.';
  } finally {
    askBtn.disabled = false;
  }
};

autoBtn.onclick = async () => {
  autoBtn.disabled = true;
  const reportEl = document.getElementById('report');
  reportEl.innerHTML = '<p>Generando reporte...</p>';
  try {
    const r = await fetch('/auto-report', { headers: authHeader() });
    const data = await r.json();
    renderReport(data);
  } catch (err) {
    reportEl.textContent = 'Error de red al generar reporte.';
  } finally {
    autoBtn.disabled = false;
  }
};

function authHeader() {
  const tok = localStorage.getItem('token');
  return tok ? { 'Authorization': 'Bearer ' + tok } : {};
}

function renderReport(data) {
  const reportEl = document.getElementById('report');
  reportEl.innerHTML = '';
  if (!data || !data.years) {
    reportEl.textContent = data && data.message ? data.message : 'No hay datos.';
    return;
  }
  // Crear contenedores
  const chartsDiv = document.createElement('div'); chartsDiv.id = 'charts';
  const kpiDiv = document.createElement('div'); kpiDiv.id = 'kpi-cards';
  reportEl.appendChild(chartsDiv);
  reportEl.appendChild(kpiDiv);

  // === Gráficos ===
  // 1. Ingresos
  const c1 = document.createElement('canvas'); chartsDiv.appendChild(c1);
  new Chart(c1.getContext('2d'), {
    type: 'bar',
    data: { labels: data.years, datasets: [{ label: 'Ingresos / Revenue', data: data.revenue }] },
    options: { plugins:{ title:{ display:true, text:'Ingresos por Año / Revenue by Year' } }, scales:{ x:{ title:{ display:true, text:'Año / Year' } }, y:{ title:{ display:true, text:'Ingresos (unidades monetarias)' } } } }
  });
  // 2. EBITDA
  const c2 = document.createElement('canvas'); chartsDiv.appendChild(c2);
  new Chart(c2.getContext('2d'), {
    type: 'line',
    data: { labels: data.years, datasets: [{ label: 'EBITDA', data: data.ebitda }] },
    options: { plugins:{ title:{ display:true, text:'EBITDA por Año' } }, scales:{ x:{ title:{ display:true, text:'Año / Year' } }, y:{ title:{ display:true, text:'EBITDA' } } } }
  });
  // 3. Margen EBITDA
  const c3 = document.createElement('canvas'); chartsDiv.appendChild(c3);
  new Chart(c3.getContext('2d'), {
    type: 'line',
    data: { labels: data.years, datasets: [{ label: 'Margen EBITDA %', data: data.ebitda_margin }] },
    options: { plugins:{ title:{ display:true, text:'Margen EBITDA (%)' } }, scales:{ x:{ title:{ display:true, text:'Año / Year' } }, y:{ title:{ display:true, text:'%' } } } }
  });
  // 4. Mezcla de costos
  const c4 = document.createElement('canvas'); chartsDiv.appendChild(c4);
  new Chart(c4.getContext('2d'), {
    type: 'pie',
    data: { labels: data.cost_labels, datasets: [{ data: data.cost_mix }] },
    options: { plugins:{ title:{ display:true, text:'Mezcla de Costos / Cost Mix' } } }
  });

  // === Tarjetas KPI ===
  data.kpis.forEach(k => {
    const li = document.createElement('div');
    let cls = 'kpi';
    const v = k.value;
    if (typeof k.good === 'number' && typeof k.warn === 'number') {
      if (v >= k.good) cls += ' ok';
      else if (v >= k.warn) cls += ' warn';
      else cls += ' bad';
    }
    li.className = cls;
    li.innerHTML = `<b>${k.name_es} / ${k.name_en}:</b> ${v}${k.unit || ''}<br><small>Fórmula: ${k.formula}</small><br><small>Evaluación: ${v >= k.good ? 'Bueno' : v >= k.warn ? 'Promedio' : 'Malo'}</small><br><small>Acción sugerida: ${v >= k.good ? k.action_good : v >= k.warn ? k.action_warn : k.action_bad}</small>`;
    kpiDiv.appendChild(li);
  });

  // === Glosario ===
  if (data.glossary) {
    const glDiv = document.createElement('div');
    glDiv.style.marginTop = '1rem';
    glDiv.innerHTML = '<h3>Glosario</h3>';
    const ul = document.createElement('ul');
    for (const [abbr, desc] of Object.entries(data.glossary)) {
      const li = document.createElement('li');
      li.innerHTML = `<b>${abbr}</b>: ${desc}`;
      ul.appendChild(li);
    }
    glDiv.appendChild(ul);
    reportEl.appendChild(glDiv);
  }
}
</script>
</body></html>
"""

# ================ Warmup: carga del modelo al arranque ================

@app.on_event("startup")
def _warm_start():
    def _preload():
        try:
            print(">> Precargando modelo de embeddings...")
            get_model()
            print(">> Modelo listo.")
        except Exception as e:
            print("!! Error precargando modelo:", e)
    Thread(target=_preload, daemon=True).start()

# Ruta manual para forzar el precargado desde navegador
@app.get("/__warmup")
def __warmup():
    get_model()
    return {"ok": True, "msg": "modelo listo"}


# ================ Mercado Pago: checkout y cupón =================
mp = mercadopago.SDK(settings.MP_ACCESS_TOKEN) if settings else None

COUPONS = {
    "INVESTU-100": {"type": "free", "desc": "Acceso gratis"},
    "INVESTU-50": {"type": "percent", "value": 50, "desc": "50% OFF"},
}


def _require_mp():
    if not settings:
        raise HTTPException(500, f"Config faltante: {SETTINGS_ERROR or 'sin settings'}")
    if not getattr(settings, "MP_ACCESS_TOKEN", None):
        raise HTTPException(500, "MP_ACCESS_TOKEN no configurado")
    if mp is None:
        raise HTTPException(500, "SDK de Mercado Pago no inicializado")


def _currency_decimals(code: str) -> int:
    return 2 if code.upper() in {"USD", "EUR", "GBP", "CLP"} else 2


@app.post("/mp/create-preference")
async def mp_create_preference(request: Request):
    """Crea una preferencia de pago y maneja cupones de descuento."""
    _require_mp()
    data = await request.json()
    firstname = (data.get("firstname") or "").strip()
    lastname = (data.get("lastname") or "").strip()
    gmail = (data.get("gmail") or "").strip().lower()
    coupon = (data.get("coupon") or "").strip().upper()

    if not gmail.endswith("@gmail.com"):
        return JSONResponse({"ok": False, "message": "Debe ingresar un Gmail válido."}, status_code=400)

    if not firstname or not lastname:
        return JSONResponse({"ok": False, "message": "Debe ingresar nombre y apellido."}, status_code=400)

    # Verificar si hay cupón válido
    c = COUPONS.get(coupon)
    price = settings.MP_PRICE_1DAY
    desc = ""
    if c:
        if c.get("type") == "free":
            # acceso libre: omitimos MP y damos token directamente
            uid = make_user_id(gmail)
            token = make_token(uid, 24)
            return {"ok": True, "free": True, "token": token, "coupon": coupon}
        if c.get("type") == "percent":
            pct = float(c.get("value", 0))
            decimals = _currency_decimals(settings.MP_CURRENCY)
            price = max(1.0, round(price * (100 - pct) / 100, decimals))
            desc = c.get("desc", "")

    # Crear preferencia
    preference_data = {
        "items": [{"title": settings.APP_NAME, "quantity": 1, "unit_price": float(price), "currency_id": settings.MP_CURRENCY}],
        "payer": {"email": gmail, "name": firstname, "surname": lastname},
        "back_urls": {
            "success": f"{settings.BASE_URL}/mp/return",  # la misma ruta maneja success/pending
            "pending": f"{settings.BASE_URL}/mp/return",
            "failure": f"{settings.BASE_URL}/mp/return",
        },
        "auto_return": "approved",
    }
    res = mp.preference().create(preference_data)
    init_point = res.get("response", {}).get("init_point")
    if not init_point:
        raise HTTPException(500, "No se pudo crear preferencia de pago")
    return {"ok": True, "free": False, "init_point": init_point, "price": price, "discount": desc}


@app.get("/mp/return")
async def mp_return(request: Request):
    """Procesa el retorno de Mercado Pago y verifica el pago."""
    _require_mp()
    payment_id = request.query_params.get("payment_id")
    status = request.query_params.get("status")
    gmail = request.query_params.get("payer_email") or ""
    coupon = request.query_params.get("coupon") or ""

    # Si se usó cupón free retornamos token aunque status no venga
    if coupon and COUPONS.get(coupon, {}).get("type") == "free":
        uid = make_user_id(gmail)
        token = make_token(uid, 24)
        html = f"<script>localStorage.setItem('token','{token}'); location.href='/portal';</script>"
        return HTMLResponse(html)

    if not payment_id:
        return HTMLResponse("<h3>Pago no recibido.</h3>")

    # Verificar pago mediante API
    try:
        payment = mp.payment().get(payment_id)
        payment_status = payment.get("response", {}).get("status", "")
        payer = payment.get("response", {}).get("payer", {})
        payer_email = (payer.get("email") or "").lower()
    except Exception:
        payment_status = status
        payer_email = gmail.lower()

    if payment_status not in {"approved", "authorized"}:
        return HTMLResponse("<h3>Pago pendiente o no aprobado.</h3>")

    if not payer_email.endswith("@gmail.com"):
        return HTMLResponse("<h3>El Gmail del pagador no es válido.</h3>")

    uid = make_user_id(payer_email)
    ensure_dirs(user_dir(uid))
    token = make_token(uid, 24)
    # Guardamos token y redirigimos al portal
    html = f"<script>localStorage.setItem('token','{token}'); location.href='/portal';</script>"
    return HTMLResponse(html)


# ================ Rutas protegidas (portal, upload, ask, auto-report) =================

@app.get("/portal", response_class=HTMLResponse)
async def portal(request: Request):
    """Sirve la página del portal solo si el usuario tiene un token válido."""
    # Validamos el token (sea por header o cookie)
    uid = require_user(request)
    # Sustituimos valores en plantilla
    html = (PORTAL_HTML
            .replace("[MAX_FILES]", str(settings.MAX_UPLOAD_FILES))
            .replace("[SINGLE_MAX]", str(settings.SINGLE_FILE_MAX_MB))
            .replace("[TOTAL_MAX]", str(settings.MAX_TOTAL_MB))
            .replace("{APP_NAME}", APP_NAME_SAFE))
    return HTMLResponse(html)


def index_worker(base_dir: str, filenames: List[str]):
    """Tarea de segundo plano: abre PDFs y agrega sus fragmentos al índice."""
    try:
        base = Path(base_dir)
        pdfs = [base / "docs" / fn for fn in filenames]
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
    """Recibe uno o varios archivos PDF y los guarda/incrementa el índice."""
    uid = require_user(request)
    base = user_dir(uid); ensure_dirs(base)

    max_files = settings.MAX_UPLOAD_FILES
    single_max = settings.SINGLE_FILE_MAX_MB * 1024 * 1024
    total_max = settings.MAX_TOTAL_MB * 1024 * 1024

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
        content = await f.read()
        total_size += len(content)
        if len(content) > single_max:
            return {"ok": False, "message": f"{name}: supera {settings.SINGLE_FILE_MAX_MB} MB"}
        with open(base / "docs" / name, "wb") as out:
            out.write(content)
        saved_names.append(name)

    if total_size > total_max:
        # elimina archivos recién guardados
        for nm in saved_names:
            try:
                (base / "docs" / nm).unlink(missing_ok=True)
            except Exception:
                pass
        return {"ok": False, "message": f"Superaste el total permitido ({settings.MAX_TOTAL_MB} MB por subida)."}

    # Indexación en segundo plano
    background_tasks.add_task(index_worker, str(base), saved_names)

    # Mensaje amigable para mostrar al usuario
    return {
        "ok": True,
        "saved": saved_names,
        "message": f"Estamos cargando tus archivo(s) PDF: {', '.join('"'+n+'"' for n in saved_names)}. Analizándolos en este instante.",
    }


@app.post("/ask")
async def ask(request: Request):
    """Responde hasta 5 preguntas utilizando el índice RAG del usuario."""
    uid = require_user(request)
    base = user_dir(uid); ensure_dirs(base)
    body = await request.json()
    questions: List[str] = body.get("questions") or []
    if not isinstance(questions, list):
        return JSONResponse({"ok": False, "message": "Formato de preguntas inválido."}, status_code=400)
    if len(questions) == 0:
        return []
    if len(questions) > 5:
        return JSONResponse({"ok": False, "message": "Máximo 5 preguntas."}, status_code=400)
    responses: List[Dict[str, str]] = []
    for q in questions:
        qs = q.strip()
        if not qs:
            responses.append({"question": qs, "answer": "Pregunta vacía."})
            continue
        yrs = parse_years(qs)
        hits = []
        # Si hay años, hacemos una búsqueda para cada uno; si no, una sola
        queries = [qs] if not yrs else [f"{qs} {y}" for y in yrs]
        for qq in queries:
            hits.extend(search_faiss(base, qq, settings.TOP_K))
        # Filtrar por año si corresponde
        if yrs:
            hits = [h for h in hits if (h[1].get('year') in yrs) or any(str(y) in h[0] for y in yrs)]
        # Ordenar por score descendente
        hits = sorted(hits, key=lambda x: x[2], reverse=True)
        # Umbral mínimo (descartar muy bajas similitudes)
        hits = [h for h in hits if h[2] >= 0.25]
        if not hits:
            responses.append({"question": qs, "answer": "No se encontró evidencia suficiente en tus PDFs para esta consulta."})
            continue
        # Agrupar por año
        grouped: Dict[int | None, List[str]] = {}
        for txt, meta, score in hits:
            y = meta.get("year") or guess_year_from_text(txt)
            grouped.setdefault(y, []).append(txt)
        # Construir respuesta
        parts: List[str] = []
        for y in sorted(grouped.keys(), key=lambda x: (x is None, x)):
            texts = grouped[y][:2]  # tomamos hasta 2 fragmentos por año
            snippet = '\n'.join([t[:400] + ('…' if len(t) > 400 else '') for t in texts])
            header = f"Para el año {y}:" if y else "General:";
            parts.append(f"<b>{header}</b> {snippet}")
        responses.append({"question": qs, "answer": "<br>".join(parts)})
    return responses


@app.get("/auto-report")
async def auto_report(request: Request):
    """Devuelve datos para construir un reporte automático con gráficos y KPI."""
    uid = require_user(request)
    # Datos de ejemplo; en un sistema real se derivan de los PDFs
    years = [2020, 2021, 2022, 2023]
    revenue = [120, 135, 150, 180]  # ingresos
    ebitda = [18, 22, 25, 28]       # EBITDA
    ebitda_margin = [round(eb/rv*100, 1) for eb, rv in zip(ebitda, revenue)]
    cost_labels = ['Fijos / Fixed', 'Variables / Variable', 'Otros / Other']
    cost_mix = [40, 50, 10]
    # KPI de ejemplo con semáforos
    kpis = [
        {
            "name_es": "Margen EBITDA",
            "name_en": "EBITDA Margin",
            "value": ebitda_margin[-1],
            "unit": "%",
            "formula": "EBITDA / Ingresos",
            "good": 20,
            "warn": 12,
            "action_good": "Mantener disciplina de costos.",
            "action_warn": "Revisar precios y gastos.",
            "action_bad": "Implementar plan de eficiencia y renegociar insumos.",
        },
        {
            "name_es": "Liquidez Corriente",
            "name_en": "Current Ratio",
            "value": 1.8,
            "unit": "",
            "formula": "Activo Corriente / Pasivo Corriente",
            "good": 2.0,
            "warn": 1.5,
            "action_good": "Liquidez adecuada.",
            "action_warn": "Monitorear ciclo de caja.",
            "action_bad": "Ajustar estructura de pasivos y mejorar cobranza.",
        },
        {
            "name_es": "Deuda / EBITDA",
            "name_en": "Debt / EBITDA",
            "value": 3.5,
            "unit": "",
            "formula": "Deuda Total / EBITDA",
            "good": 3.0,
            "warn": 4.0,
            "action_good": "Apalancamiento bajo control.",
            "action_warn": "Revisar plan de amortización.",
            "action_bad": "Reducir deuda o incrementar EBITDA rápidamente.",
        },
    ]
    glossary = {
        "EBITDA": "Ganancias antes de Intereses, Impuestos, Depreciación y Amortización / Earnings Before Interest, Taxes, Depreciation and Amortization",
        "ROA": "Retorno sobre Activos / Return on Assets",
        "ROE": "Retorno sobre Patrimonio / Return on Equity",
        "WACC": "Costo Promedio Ponderado de Capital / Weighted Average Cost of Capital",
        "FNE": "Flujo Neto de Efectivo / Net Cash Flow",
    }
    return {
        "years": years,
        "revenue": revenue,
        "ebitda": ebitda,
        "ebitda_margin": ebitda_margin,
        "cost_labels": cost_labels,
        "cost_mix": cost_mix,
        "kpis": kpis,
        "glossary": glossary,
    }


@app.get("/health")
def health():
    """Devuelve OK si la configuración está cargada correctamente."""
    if SETTINGS_ERROR:
        return {"ok": False, "error": SETTINGS_ERROR}
    return {"ok": True}
