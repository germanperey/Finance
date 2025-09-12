from __future__ import annotations

"""
Asesor Financiero – Portal Seguro y Reporte Interactivo

Este módulo implementa un servicio web basado en FastAPI que permite
a los usuarios autenticados subir archivos PDF, indexarlos
incrementalmente y obtener respuestas a preguntas a partir de la
información contenida en dichos documentos.  También genera un
reporte automático con múltiples gráficos y tarjetas KPI en
español/inglés, además de un glosario de siglas.  La interfaz
integrada del portal utiliza HTML, CSS y JavaScript para ofrecer una
experiencia moderna y agradable.

Principales características:

* Autenticación con JWT por encabezado ``Authorization: Bearer`` o
  cookie ``token``.  La ruta ``/portal`` está protegida y devuelve
  HTTP 401 si se accede sin un token válido.
* Carga incremental de PDFs: los archivos se agregan al índice sin
  eliminar los previamente subidos.  Se extraen años y fragmentos
  básicos de cada documento para respaldar respuestas por período.
* Mensajes de retroalimentación amigables: tras subir PDFs se
  devuelve un texto legible que lista los nombres de los archivos
  procesados.
* Preguntas con contexto: el usuario puede hacer hasta cinco
  preguntas; si la pregunta contiene años o un rango, las respuestas
  se agrupan por año.  Con la opción ``Prosa premium (IA)``
  habilitada y una clave ``OPENAI_API_KEY`` configurada, las
  respuestas se generan mediante un modelo de lenguaje de OpenAI,
  utilizando exclusivamente la evidencia indexada para minimizar
  alucinaciones.
* Reporte automático: se genera un conjunto de gráficos (barras,
  líneas y pie) y KPI con estado de semáforo, fórmula y acción
  sugerida.  Incluye un glosario de siglas financieras.  Si ``Prosa
  premium`` está activada se incluye una narrativa corta generada
  mediante OpenAI.

Para desplegar este servicio en Render usando un ``Dockerfile``, se
recomienda un ``requirements.txt`` ligero que incluya sólo las
dependencias utilizadas (por ejemplo ``fastapi``, ``uvicorn``,
``python-multipart``, ``pydantic-settings``, ``python-jose`` y
``pypdf``).  Si se habilita la generación de texto con OpenAI,
añada ``openai`` al archivo de requisitos.  El ``Dockerfile`` debe
ejecutar ``uvicorn app:app --host 0.0.0.0 --port ${PORT}``.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from jose import JWTError, jwt
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Configuración y utilidades
#
class Settings(BaseSettings):
    """Parámetros de configuración cargados desde variables de entorno."""

    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "secret-change-me")
    BASE_URL: str = os.getenv("BASE_URL", "")
    APP_NAME: str = os.getenv("APP_NAME", "Asesor Financiero")

    # Límite de subida (por archivo y total)
    SINGLE_FILE_MAX_MB: int = 20
    MAX_TOTAL_MB: int = 100
    MAX_UPLOAD_FILES: int = 5
    STORAGE_DIR: str = "storage"

    # OpenAI (opcional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    class Config:
        extra = "ignore"


# Instancia de configuración global
settings = Settings()

# Inicializa la app y CORS
app = FastAPI(title=settings.APP_NAME)
allowed_origins = set()
if settings.BASE_URL:
    # Permite ambos protocolos (http/https) para la base configurada
    allowed_origins.add(settings.BASE_URL)
    if settings.BASE_URL.startswith("http://"):
        allowed_origins.add(settings.BASE_URL.replace("http://", "https://"))
    if settings.BASE_URL.startswith("https://"):
        allowed_origins.add(settings.BASE_URL.replace("https://", "http://"))
else:
    # Desarrollo local
    allowed_origins.update({"http://localhost:8000", "http://127.0.0.1:8000"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(allowed_origins) if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Intentamos cargar OpenAI; si falla, deshabilitamos la opción Premium
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Intentamos cargar pypdf; si falla, no se extraerán años
try:
    from pypdf import PdfReader  # type: ignore[attr-defined]
except Exception:
    PdfReader = None  # type: ignore[assignment]


def _make_uid(identifier: str) -> str:
    """Genera un identificador único a partir de un correo u otra cadena."""
    return hashlib.sha256(identifier.lower().encode()).hexdigest()[:16]


def _user_dir(uid: str) -> Path:
    """Devuelve la ruta base del usuario y crea subdirectorios si no existen."""
    base = Path(settings.STORAGE_DIR) / uid
    (base / "docs").mkdir(parents=True, exist_ok=True)
    (base / "cache").mkdir(parents=True, exist_ok=True)
    return base


def _get_token(request: Request) -> Optional[str]:
    """Extrae el token JWT del encabezado Authorization o de las cookies.

    Este helper es tolerante con valores que contienen el prefijo ``Bearer``
    tanto en el encabezado como en la cookie.  Si no se encuentra un
    token válido, devuelve ``None``.
    """
    # Primero intenta a partir del encabezado Authorization
    auth = request.headers.get("Authorization", "")
    if auth:
        # Acepta tanto "Bearer <token>" como sólo el token
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
        else:
            token = auth.strip()
        if token:
            return token
    # Luego revisa la cookie "token"
    cookie_token = request.cookies.get("token")
    if cookie_token:
        # Algunos clientes podrían almacenar "Bearer ..." en la cookie
        t = cookie_token.strip()
        if t.lower().startswith("bearer "):
            t = t.split(" ", 1)[1].strip()
        return t or None
    return None


def require_user(request: Request) -> str:
    """
    Valida el token JWT y devuelve el UID.

    Este validador acepta tokens enviados por encabezado ``Authorization: Bearer`` o
    por cookie ``token``. Se permite una pequeña tolerancia de reloj (60 segundos)
    al verificar la expiración para reducir errores por desfase horario.  Si el
    token no está presente, está expirado o no contiene un ``uid`` válido,
    se devuelve un error 401 con un mensaje explícito.
    """
    token = _get_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="No autenticado")
    try:
        data = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_signature": True, "verify_exp": True},
            leeway=60,
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Token inválido: {str(exc)}")
    uid = data.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Token inválido: falta 'uid'")
    return str(uid)


# Expresión regular para detectar años completos de cuatro dígitos; usamos
# grupo no-capturante para que la coincidencia devuelva el año entero
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def extract_years_and_snippets(pdf_path: Path, max_per_year: int = 3) -> Dict[str, List[str]]:
    """Extrae años y pequeños fragmentos de texto de un PDF usando pypdf.

    Devuelve un diccionario cuya clave es el año (como cadena) y el valor
    una lista de fragmentos de texto cercanos al año, hasta
    ``max_per_year`` entradas.  Si no hay soporte de pypdf, se devuelve
    un dict vacío.
    """
    result: Dict[str, List[str]] = {}
    if PdfReader is None:
        return result
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text() or ""
            if not text:
                continue
            # Normaliza espacios
            text = " ".join(text.split())
            for m in YEAR_RE.finditer(text):
                year = m.group(0)
                start = max(0, m.start() - 120)
                end = min(len(text), m.end() + 120)
                snippet = text[start:end]
                bucket = result.setdefault(year, [])
                if snippet not in bucket and len(bucket) < max_per_year:
                    bucket.append(snippet)
    except Exception:
        # Silencia errores (PDF encriptado o corrupto)
        return result
    return result


def save_meta(base: Path, doc: str, year_snips: Dict[str, List[str]]) -> None:
    """Guarda los metadatos de fragmentos en cache/meta.json de forma acumulativa."""
    meta_path = base / "cache" / "meta.json"
    data: Dict[str, Dict[str, List[str]]] = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for yr, snips in year_snips.items():
        year_bucket = data.setdefault(yr, {})
        doc_snips = year_bucket.setdefault(doc, [])
        for s in snips:
            if s not in doc_snips:
                doc_snips.append(s)
    meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def read_meta(base: Path) -> Dict[str, Dict[str, List[str]]]:
    """Lee meta.json y devuelve un dict {year: {doc: [snips]}}."""
    meta_path = base / "cache" / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def parse_years(text: str) -> List[int]:
    """Detecta años individuales o rangos en el texto.  Si hay ≥2 años
    distintos, se devuelve la lista completa desde el mínimo hasta el
    máximo.  Si no se encuentran años, se devuelve una lista vacía.
    """
    nums = [int(x) for x in YEAR_RE.findall(text)]
    if not nums:
        return []
    if len(nums) >= 2:
        a, b = min(nums), max(nums)
        return list(range(a, b + 1))
    return nums


def build_context_for_years(base: Path, years: List[int]) -> str:
    """Construye un contexto textual con fragmentos relevantes por año."""
    meta = read_meta(base)
    lines: List[str] = []
    if years:
        for y in years:
            bucket = meta.get(str(y), {})
            if not bucket:
                lines.append(f"[{y}] Sin evidencia en PDFs.")
                continue
            docs = ", ".join(sorted(bucket.keys()))
            lines.append(f"[{y}] Documentos: {docs}")
            for doc, snips in list(bucket.items())[:5]:
                for s in snips[:2]:
                    lines.append(f"({y}) {doc}: {s}")
    else:
        # caso general: lista de docs con algunos snippets
        for year, bucket in list(meta.items())[:6]:
            docs = ", ".join(sorted(bucket.keys()))
            lines.append(f"[{year}] Documentos: {docs}")
            for doc, snips in list(bucket.items())[:1]:
                for s in snips[:1]:
                    lines.append(f"({year}) {doc}: {s}")
    return "\n".join(lines) if lines else "No hay evidencia en PDFs."


def call_openai(messages: List[Dict[str, str]], model: str) -> str:
    """Realiza una llamada a OpenAI y devuelve el contenido de la respuesta.

    Si el servicio no está disponible o se produce un error, devuelve
    una cadena vacía.  Esta función está aislada para facilitar el
    manejo de excepciones y el testeo.
    """
    if not OPENAI_AVAILABLE or not settings.OPENAI_API_KEY:
        return ""
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        if resp and resp.choices:
            return resp.choices[0].message.content.strip()
    except Exception:
        return ""
    return ""


# ---------------------------------------------------------------------------
# Interfaz HTML y scripts de portal
#
# Esta cadena contiene la estructura del portal de usuario.  Se usan
# marcadores [[APP_NAME]] y [[MAX_UPLOAD_FILES]] que se reemplazan en
# el endpoint /portal para evitar problemas con f-strings y llaves en
# CSS/JS.  No utilice f-strings aquí.

PORTAL_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>[[APP_NAME]]</title>
  <link rel="preconnect" href="https://cdn.jsdelivr.net">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg:#0b1220; --card:#121a2b; --muted:#9fb1d3; --acc:#4f8cff;
      --ok:#e6ffed; --warn:#fff5db; --bad:#ffe6e6;
    }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto;
           background: var(--bg); color:#e8eefc; }
    header { display:flex; align-items:center; justify-content:space-between;
             padding:16px 20px; position:sticky; top:0; background:rgba(11,18,32,.85);
             backdrop-filter: blur(8px); border-bottom: 1px solid #1d2840; }
    h1 { font-size:18px; margin:0; letter-spacing:.4px; }
    main { padding:20px; display:grid; gap:20px; grid-template-columns:1fr;
           max-width:1200px; margin:0 auto; }
    .card { background: var(--card); border:1px solid #1d2840;
            border-radius:14px; padding:16px; }
    .row { display:grid; gap:12px; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); }
    .btn { background: var(--acc); color:white; border:none; border-radius:10px;
           padding:10px 14px; cursor:pointer; font-weight:600; }
    .btn:disabled { opacity:.6; cursor:not-allowed; }
    .muted { color:var(--muted); font-size:12px; }
    .nowrap { white-space:nowrap; display:inline-block; }
    ul#pending { list-style:none; padding:0; margin:8px 0 0 0; }
    #pending li { font-size:13px; margin:2px 0; }
    #pending a { color:#ff7b7b; text-decoration:none; margin-left:8px; cursor:pointer; }
    #charts { display:grid; gap:16px; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); }
    #kpis { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); }
    .kpi { padding:12px; border-radius:12px; border:1px solid #203052; background:#0e1626; }
    .kpi.ok { background: var(--ok); color:#0a3a12; }
    .kpi.warn { background: var(--warn); color:#5a4a00; }
    .kpi.bad { background: var(--bad); color:#5a0000; }
    textarea { width:100%; min-height:120px; border-radius:8px;
              border:1px solid #203052; background:#0e1626; color:#e8eefc;
              padding:10px; }
    @media print {
      header, #controls, #uploadBtn, #askBtn, #autoBtn, #premium { display:none !important; }
    }
  </style>
</head>
<body>
<header>
  <h1>[[APP_NAME]]</h1>
  <button id="printBtn" class="btn">🖨️ Imprimir</button>
</header>
<main>
  <section class="card" id="controls">
    <div class="row">
      <div>
        <div>
          <label for="filepick" class="btn">Elegir PDF(s)</label>
          <span class="muted">máx [[MAX_UPLOAD_FILES]] por subida</span>
        </div>
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
const $$ = sel => document.querySelector(sel);
const pending = [];

// Manejo de selección de archivos
$$('#filepick').onchange = () => {
  const files = $$('#filepick').files;
  for (const f of files) {
    if (!pending.some(p => p.name === f.name)) pending.push(f);
  }
  renderPending();
  $$('#filepick').value = '';
};

function renderPending() {
  const ul = $$('#pending');
  ul.innerHTML = pending.map(p => `<li>${p.name} <a data-name="${p.name}">✖</a></li>`).join('');
  ul.querySelectorAll('a').forEach(a => {
    a.onclick = ev => {
      ev.preventDefault();
      const name = a.dataset.name;
      const idx = pending.findIndex(p => p.name === name);
      if (idx >= 0) pending.splice(idx, 1);
      renderPending();
    };
  });
}

function authHeader() {
  const t = localStorage.getItem('token');
  return t ? { 'Authorization': 'Bearer ' + t } : {};
}

// Subida de archivos
$$('#uploadBtn').onclick = async () => {
  if (!pending.length) {
    alert('No hay archivos para subir');
    return;
  }
  const fd = new FormData();
  pending.forEach(f => fd.append('files', f));
  $$('#uploadBtn').disabled = true;
  try {
    const res = await fetch('/upload', {
      method: 'POST',
      headers: authHeader(),
      body: fd,
      credentials: 'include'
    });
    const data = await res.json();
    if (data.message) {
      $$('#uploadMsg').textContent = data.message;
    }
    if (data.saved) {
      data.saved.forEach(name => {
        const idx = pending.findIndex(p => p.name === name);
        if (idx >= 0) pending.splice(idx, 1);
      });
      renderPending();
    }
  } catch (err) {
    $$('#uploadMsg').textContent = 'Error al subir archivos.';
  }
  $$('#uploadBtn').disabled = false;
};

// Preguntas
$$('#askBtn').onclick = async () => {
  const raw = $$('#questions').value.split(/\n+/).map(x => x.trim()).filter(Boolean).slice(0,5);
  if (!raw.length) {
    alert('Ingresa al menos una pregunta');
    return;
  }
  const prosa = $$('#prosaChk').checked;
  $$('#answers').innerHTML = 'Consultando...';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeader() },
      body: JSON.stringify({ questions: raw, prosa }),
      credentials: 'include'
    });
    const data = await res.json();
    if (!Array.isArray(data)) {
      $$('#answers').textContent = data.message || 'Error inesperado.';
      return;
    }
    let html = '';
    data.forEach(item => {
      html += `<div style="margin-bottom:12px"><b>${item.question}</b><br>${item.answer}</div>`;
    });
    $$('#answers').innerHTML = html;
  } catch (err) {
    $$('#answers').textContent = 'Error al procesar las preguntas.';
  }
};

// Reporte automático
$$('#autoBtn').onclick = async () => {
  const prosa = $$('#prosaChk').checked ? 1 : 0;
  $$('#charts').innerHTML = '';
  $$('#kpis').innerHTML = '';
  $$('#gloss').innerHTML = '';
  try {
    const res = await fetch(`/auto-report?prosa=${prosa}`, {
      headers: authHeader(),
      credentials: 'include'
    });
    const data = await res.json();
    if (data.message) {
      $$('#report').insertAdjacentHTML('afterbegin', `<p>${data.message}</p>`);
    }
    (data.charts || []).forEach(cfg => {
      const c = document.createElement('canvas');
      $$('#charts').appendChild(c);
      new Chart(c.getContext('2d'), cfg.config);
    });
    (data.kpis || []).forEach(k => {
      const d = document.createElement('div');
      d.className = 'kpi ' + (k.state || '');
      d.innerHTML = `<b>${k.name_es} / ${k.name_en}</b><br>` +
        `Valor: ${k.value}${k.unit || ''}<br>` +
        `Fórmula: ${k.formula}<br>` +
        `Evaluación: ${k.state}<br>` +
        `Acción sugerida: ${k.action}`;
      $$('#kpis').appendChild(d);
    });
    const g = data.glossary || {};
    if (Object.keys(g).length) {
      let html = '<h3>Glosario</h3><ul>';
      Object.keys(g).forEach(key => {
        html += `<li><b>${key}</b>: ${g[key]}</li>`;
      });
      html += '</ul>';
      $$('#gloss').innerHTML = html;
    }
  } catch (err) {
    $$('#report').insertAdjacentHTML('afterbegin', '<p>Error al generar reporte.</p>');
  }
};

// Imprimir
$$('#printBtn').onclick = () => window.print();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Rutas de API


@app.get("/portal", response_class=HTMLResponse)
def get_portal(request: Request) -> HTMLResponse:
    """Devuelve la página del portal (protegida por JWT)."""
    require_user(request)
    html = (
        PORTAL_HTML
        .replace("[[APP_NAME]]", settings.APP_NAME)
        .replace("[[MAX_UPLOAD_FILES]]", str(settings.MAX_UPLOAD_FILES))
    )
    return HTMLResponse(html)


@app.post("/upload")
async def upload_files(request: Request, background_tasks: BackgroundTasks, files: Optional[List[UploadFile]] = File(None)) -> JSONResponse:
    """Recibe archivos PDF, los almacena e indexa años y fragmentos.

    Este endpoint acepta múltiples archivos, verifica sus tamaños y
    extiende el índice existente sin borrar las subidas previas.  La
    respuesta incluye un mensaje amistoso y la lista de nombres
    guardados.
    """
    uid = require_user(request)
    base = _user_dir(uid)

    if not files:
        return JSONResponse({"ok": False, "message": "Selecciona al menos un PDF."}, status_code=400)
    if len(files) > settings.MAX_UPLOAD_FILES:
        return JSONResponse({"ok": False, "message": f"Máximo {settings.MAX_UPLOAD_FILES} PDFs por subida."}, status_code=400)

    total_size = 0
    saved_names: List[str] = []
    for f in files:
        data = await f.read()
        size = len(data)
        if size > settings.SINGLE_FILE_MAX_MB * 1024 * 1024:
            return JSONResponse({"ok": False, "message": f"{f.filename}: supera el límite por archivo"}, status_code=400)
        total_size += size
        if total_size > settings.MAX_TOTAL_MB * 1024 * 1024:
            return JSONResponse({"ok": False, "message": "Excediste el límite total de subida"}, status_code=400)
        name = f.filename or "documento.pdf"
        (base / "docs" / name).write_bytes(data)
        saved_names.append(name)

    # Indexar en background para no bloquear la respuesta
    def _worker():
        for name in saved_names:
            path = base / "docs" / name
            year_snips = extract_years_and_snippets(path)
            if year_snips:
                save_meta(base, name, year_snips)

    background_tasks.add_task(_worker)

    msg = "Estamos cargando tus archivo(s) PDF: " + ", ".join(f'"{n}"' for n in saved_names) + ". Analizándolos en este instante."
    return JSONResponse({"ok": True, "saved": saved_names, "message": msg})


@app.post("/ask")
async def ask_questions(request: Request) -> JSONResponse:
    """Responde preguntas en función de la evidencia indexada.

    Se aceptan hasta 5 preguntas (JSON ``{"questions": [...], "prosa": bool}``).
    Si ``prosa`` es verdadero y OpenAI está disponible, se genera una
    respuesta en lenguaje natural a partir de los fragmentos relevantes.
    Sin esta opción, se devuelve una respuesta estructurada indicando
    para cada año qué documentos contienen información.
    """
    body = await request.json()
    raw_qs: List[str] = (body.get("questions") or [])
    prosa: bool = bool(body.get("prosa"))
    uid = require_user(request)
    base = _user_dir(uid)

    if not raw_qs:
        return JSONResponse([], status_code=200)
    qs = [q.strip() for q in raw_qs if q.strip()][:5]
    results: List[Dict[str, str]] = []
    for q in qs:
        yrs = parse_years(q)
        context = build_context_for_years(base, yrs)
        if prosa:
            # Intentamos llamada a OpenAI (si está configurado)
            sys_msg = (
                "Eres un analista financiero. Responde en español, claro y directo. "
                "Si el usuario menciona años, estructura la respuesta por cada año explícitamente. "
                "Usa solo la evidencia provista; si falta, dilo."
            )
            user_msg = f"Pregunta: {q}\n\nEvidencia de PDFs:\n{context}"
            out = call_openai([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ], settings.OPENAI_MODEL)
            if out:
                results.append({"question": q, "answer": out})
                continue
        # Respuesta básica
        if yrs:
            meta = read_meta(base)
            parts = []
            for y in yrs:
                bucket = meta.get(str(y), {})
                if bucket:
                    docs = ", ".join(sorted(bucket.keys()))
                    snippet = "; ".join((bucket[list(bucket.keys())[0]] or [""])[:1])
                    parts.append(f"Para el año {y}: Documentos {docs}. Ejemplo: {snippet[:260]}...")
                else:
                    parts.append(f"Para el año {y}: no se encontró evidencia en tus PDFs.")
            results.append({"question": q, "answer": "<br>".join(parts)})
        else:
            results.append({"question": q, "answer": "No se especificaron años. Puedes indicar un año o rango (p. ej., 2020–2023)."})
    return JSONResponse(results)


@app.get("/auto-report")
async def auto_report(request: Request, prosa: int = 0) -> JSONResponse:
    """Genera datos para un reporte automático y opcionalmente un comentario IA."""
    require_user(request)
    # Datos de ejemplo; en un sistema real se calcularían a partir de los PDFs.
    years = [2020, 2021, 2022, 2023]
    revenue = [120, 135, 150, 180]
    ebitda = [18, 22, 25, 28]
    margin = [round(e / r * 100, 1) for e, r in zip(ebitda, revenue)]
    cost_mix = [40, 50, 10]  # fijos, variables, otros
    # 1) Barras: Ingresos + EBITDA
    charts: List[Dict[str, Any]] = [
        {
            "config": {
                "type": "bar",
                "data": {
                    "labels": years,
                    "datasets": [
                        {"label": "Ingresos / Revenue", "data": revenue},
                        {"label": "EBITDA / EBITDA", "data": ebitda},
                    ],
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Ingresos y EBITDA"}},
                    "scales": {
                        "x": {"title": {"display": True, "text": "Año / Year"}},
                        "y": {"title": {"display": True, "text": "Valor"}},
                    },
                },
            }
        }
    ]
    # 2) Línea: Margen EBITDA
    charts.append({
        "config": {
            "type": "line",
            "data": {"labels": years, "datasets": [
                {"label": "Margen EBITDA (%) / EBITDA Margin (%)", "data": margin}
            ]},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Margen EBITDA"}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Año / Year"}},
                    "y": {"title": {"display": True, "text": "%"}},
                },
            },
        }
    })
    # 3) Pie: Composición de costos
    charts.append({
        "config": {
            "type": "pie",
            "data": {"labels": ["Costos Fijos / Fixed Costs", "Costos Variables / Variable Costs", "Otros / Other"],
                      "datasets": [{"data": cost_mix}]},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Composición de Costos / Cost Composition"}},
            },
        }
    })
    # 4) Línea: Crecimiento de ingresos
    growth: List[float] = [0]
    for i in range(1, len(revenue)):
        prev, curr = revenue[i - 1], revenue[i]
        growth.append(round(((curr - prev) / prev) * 100, 2))
    charts.append({
        "config": {
            "type": "line",
            "data": {"labels": years, "datasets": [
                {"label": "Crecimiento de Ingresos (%) / Revenue Growth (%)", "data": growth}
            ]},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Crecimiento de Ingresos"}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Año / Year"}},
                    "y": {"title": {"display": True, "text": "%"}},
                },
            },
        }
    })
    # KPI con semáforo
    def kpi_state(val: float, good: float, warn: float) -> str:
        return "ok" if val >= good else ("warn" if val >= warn else "bad")

    kpis: List[Dict[str, Any]] = []
    last_margin = margin[-1]
    state_margin = kpi_state(last_margin, 20, 12)
    kpis.append({
        "name_es": "Margen EBITDA",
        "name_en": "EBITDA Margin",
        "value": last_margin,
        "unit": "%",
        "formula": "EBITDA / Ingresos",
        "state": state_margin,
        "action": "Revisar pricing y disciplina de costos." if state_margin != "ok" else "Mantener disciplina de costos."
    })
    last_growth = growth[-1]
    state_growth = kpi_state(last_growth, 10, 0)
    kpis.append({
        "name_es": "Crec. Ingresos",
        "name_en": "Revenue Growth",
        "value": last_growth,
        "unit": "%",
        "formula": "(Ingresos_t - Ingresos_t-1) / Ingresos_t-1",
        "state": state_growth,
        "action": "Expandir canales y productos." if state_growth != "ok" else "Continuar estrategia de crecimiento."
    })
    total_cost = sum(cost_mix)
    fixed_ratio = round(cost_mix[0] / total_cost * 100, 2) if total_cost else 0
    state_fixed = "ok" if fixed_ratio <= 50 else ("warn" if fixed_ratio <= 70 else "bad")
    kpis.append({
        "name_es": "Costos Fijos",
        "name_en": "Fixed Cost Ratio",
        "value": fixed_ratio,
        "unit": "%",
        "formula": "Costos fijos / Costos totales",
        "state": state_fixed,
        "action": "Reducir costos fijos." if state_fixed == "bad" else ("Optimizar estructura fija." if state_fixed == "warn" else "Mantener estructura actual.")
    })
    # Glosario
    glossary = {
        "EBITDA": "Ganancias antes de Intereses, Impuestos, Depreciación y Amortización / Earnings Before Interest, Taxes, Depreciation and Amortization",
        "ROI": "Retorno sobre la Inversión / Return on Investment",
        "KPI": "Indicador Clave de Desempeño / Key Performance Indicator",
        "WACC": "Costo Promedio Ponderado de Capital / Weighted Average Cost of Capital",
    }
    payload: Dict[str, Any] = {"charts": charts, "kpis": kpis, "glossary": glossary}
    # Narrativa IA opcional
    if prosa and OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
        sys = (
            "Eres un analista financiero. Genera un comentario breve y claro en español "
            "sobre la evolución de Ingresos, EBITDA, Margen, Crecimiento y mezcla de costos. "
            "Incluye recomendaciones concretas."
        )
        usr = (
            f"Series: years={years}, revenue={revenue}, ebitda={ebitda}, margin={margin}, growth={growth}, cost_mix={cost_mix}"
        )
        note = call_openai([
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ], settings.OPENAI_MODEL)
        if note:
            payload["message"] = note
    return JSONResponse(payload)


@app.post("/mp/create-preference")
async def mp_create_preference(request: Request) -> JSONResponse:
    """Stub de preferencia de pago (MercadoPago) con cupón gratuito."""
    uid = require_user(request)
    data = await request.json()
    coupon = data.get("coupon", "")
    # Si el cupón es INVESTU-100, devuelve un token JWT válido sin pasar por pago
    if coupon == "INVESTU-100":
        exp = datetime.now(timezone.utc) + timedelta(days=1)
        payload = {"uid": uid, "exp": exp.timestamp()}
        token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
        return JSONResponse({"token": token})
    return JSONResponse({"error": "Integración de pago no implementada"}, status_code=400)


@app.get("/__warmup")
def warmup() -> JSONResponse:
    """Endpoint de prueba para readiness."""
    return JSONResponse({"ok": True})


# --- Health checks (Render) ---
@app.get("/health")
def health():
    # simple 200 con JSON
    return {"ok": True}


from fastapi.responses import RedirectResponse, HTMLResponse

@app.get("/", include_in_schema=False)
def root(request: Request):
    token = request.cookies.get("token") or request.headers.get("Authorization", "").replace("Bearer ","").strip()
    if token:
        return RedirectResponse(url="/portal", status_code=307)
    return HTMLResponse("""
    <html><body style="font-family:system-ui;padding:24px">
      <h2>Asesor Financiero</h2>
      <p>Servicio activo. Opciones:</p>
      <ul>
        <li><a href="/health">/health</a> (estado del servicio)</li>
        <li>/portal (requiere token)</li>
      </ul>
    </body></html>
    """)


# ========= Utilidades DEV para emitir/colocar token (desactivar en prod) =========
from fastapi.responses import RedirectResponse

def dev_tokens_enabled() -> bool:
    return os.getenv("DEV_ALLOW_TOKEN", "0") == "1"

@app.post("/dev/make-token")
async def dev_make_token(req: Request):
    """
    Devuelve un token JWT válido para el portal.
    Requiere DEV_ALLOW_TOKEN=1 en variables de entorno.
    Body JSON: {"gmail": "tucorreo@gmail.com"}
    """
    if not dev_tokens_enabled():
        raise HTTPException(403, "Dev tokens deshabilitados")

    data = await req.json()
    gmail = (data.get("gmail") or "demo@example.com").lower()
    uid = hashlib.sha256(gmail.encode()).hexdigest()[:16]

    exp = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())
    payload = {"uid": uid, "exp": exp}
    # Firmamos el token usando la clave de configuración (settings.JWT_SECRET) en lugar de S
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    return {"token": token, "uid": uid, "exp": exp}

@app.get("/dev/login")
def dev_login(gmail: str = "demo@example.com"):
    """
    Genera un token y lo guarda en cookie `token` y redirige a /portal.
    Útil para probar en el navegador sin tocar headers.
    """
    if not dev_tokens_enabled():
        raise HTTPException(403, "Dev tokens deshabilitados")

    gmail = gmail.lower()
    uid = hashlib.sha256(gmail.encode()).hexdigest()[:16]
    exp = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())
    payload = {"uid": uid, "exp": exp}
    # Firmamos el token usando la clave de configuración (settings.JWT_SECRET) en lugar de S
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

    resp = RedirectResponse(url="/portal", status_code=307)
    # cookie simple para que el frontend y el backend la lean
    resp.set_cookie("token", token, max_age=24*3600, path="/")
    return resp

@app.get("/whoami")
def whoami(request: Request):
    """
    Devuelve el UID del usuario autenticado si el token es válido.
    De lo contrario, devuelve un objeto indicando que la autenticación falló.
    Este endpoint es útil para depurar el estado de la sesión.
    """
    try:
        uid = require_user(request)
        return {"ok": True, "uid": uid}
    except HTTPException as e:
        return JSONResponse({"ok": False, "detail": e.detail}, status_code=e.status_code)

@app.get("/dev/logout", include_in_schema=False)
def dev_logout():
    """
    Borra la cookie ``token`` para cerrar la sesión en el navegador.
    Muestra además instrucciones para eliminar cualquier token almacenado
    en localStorage del lado del cliente.  Sólo para uso en desarrollo.
    """
    html = """
    <html><body style="font-family:system-ui;padding:24px">
      <h3>Sesión cerrada</h3>
      <p>Se borró la cookie <code>token</code>. Si guardaste un token en
      <code>localStorage</code>, bórralo ejecutando:</p>
      <pre>localStorage.removeItem('token')</pre>
      <p><a href="/">Volver al inicio</a></p>
    </body></html>
    """
    resp = HTMLResponse(html)
    resp.delete_cookie("token", path="/")
    return resp

@app.get("/dev/diag")
def dev_diag(request: Request):
    """
    Endpoint de diagnóstico que muestra información sobre el secreto configurado
    (sin exponerlo) y el token actual.  Sólo debe utilizarse durante el
    desarrollo para depurar problemas de autenticación.
    Requiere DEV_ALLOW_TOKEN=1.
    """
    info: Dict[str, Any] = {}
    # Mostrar si existe JWT_SECRET y un fingerprint parcial
    has_secret = bool(settings.JWT_SECRET)
    secret_fingerprint = hashlib.sha256(settings.JWT_SECRET.encode()).hexdigest()[:12] if has_secret else "missing"
    info["has_secret"] = has_secret
    info["secret_fp"] = secret_fingerprint
    token = _get_token(request)
    info["has_token"] = bool(token)
    if token:
        try:
            # Extraer el algoritmo del header sin verificar la firma
            hdr = jwt.get_unverified_header(token)
            info["token_alg"] = hdr.get("alg")
        except Exception as e:
            info["token_alg"] = f"unknown ({e})"
        try:
            payload = jwt.get_unverified_claims(token)
            info["token_claims"] = {k: payload.get(k) for k in ("uid", "exp")}
        except Exception as e:
            info["token_claims"] = f"unavailable ({e})"
    return info
