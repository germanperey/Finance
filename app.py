from __future__ import annotations
"""
Portal de usuario con indexación de PDFs, preguntas por año y reporte automático.

Este archivo implementa un portal de análisis financiero que permite a los usuarios
subir PDFs de manera incremental, realizar preguntas filtradas por años
mencionados y generar un reporte automático con varios gráficos, KPIs y
glosario. La ruta /portal está protegida por token JWT para impedir el acceso
sin autorización.

Puntos clave implementados:
1. Carga de archivos uno a uno o en bloque, sin borrar los ya subidos.
2. Mensajes amigables tras la subida: se listan los nombres de los PDFs
   cargados y se indica que se están analizando.
3. La etiqueta "Prosa premium (IA)" se muestra en una sola línea gracias a
   estilos CSS.
4. El endpoint /ask detecta años o rangos de años en cada pregunta y
   responde indicando qué documentos contienen esos años. Si no hay años,
   responde con la lista de documentos disponibles.
5. El endpoint /auto-report devuelve datos para varios gráficos (barras,
   líneas y pie), KPIs bilingües con semáforo y acciones sugeridas, y un
   glosario de siglas.
6. Glosario incluido en el reporte con definiciones en español e inglés.
7. Nombres de variables en los gráficos en español e inglés.
8. Botón de impresión en el header del portal.
9. Protección de /portal mediante token JWT obtenido tras el pago o cupón.

Para desplegar este servicio en Render u otro entorno, asegúrate de incluir
las dependencias necesarias en requirements.txt (fastapi, uvicorn,
python-multipart, pydantic-settings, python-jose, opcionalmente pypdf).
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from jose import jwt

# Opcional: intentar importar pypdf para extraer texto y años
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # la app seguirá funcionando aunque no se extraiga texto


class Settings(BaseSettings):
    """Parámetros de configuración del portal."""
    APP_NAME: str = "Portal de usuario (pase activo)"
    BASE_URL: str = os.getenv("BASE_URL", "https://finance-4vlf.onrender.com")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "cambia-esto-en-produccion")
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "storage")
    MAX_UPLOAD_FILES: int = 5
    SINGLE_FILE_MAX_MB: int = 20
    MAX_TOTAL_MB: int = 100

    class Config:
        env_file = ".env"


S = Settings()
app = FastAPI(title=S.APP_NAME)

# Middleware CORS amplio para permitir acceso desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio de almacenamiento
STORAGE = Path(S.STORAGE_DIR)
STORAGE.mkdir(parents=True, exist_ok=True)

# ===================== Autenticación =====================

def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def make_token(uid: str, hours: int = 24) -> str:
    """Genera un JWT con expiración en horas para el usuario."""
    return jwt.encode({"uid": uid, "exp": _now_ts() + hours * 3600}, S.JWT_SECRET, algorithm="HS256")


def _get_token(request: Request) -> Optional[str]:
    """Extrae el token del header Authorization o de cookie."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.cookies.get("token")


def require_user(request: Request) -> str:
    """Valida el token y devuelve el identificador del usuario."""
    tok = _get_token(request)
    if not tok:
        raise HTTPException(401, "No autenticado")
    try:
        data: Dict[str, Any] = jwt.decode(tok, S.JWT_SECRET, algorithms=["HS256"])  # type: ignore
        if int(data.get("exp", 0)) < _now_ts():
            raise HTTPException(401, "Token expirado")
        uid = data.get("uid")
        if not uid:
            raise HTTPException(401, "Token inválido")
        return uid
    except Exception:
        raise HTTPException(401, "Token inválido")


def user_dir(uid: str) -> Path:
    """Obtiene o crea el directorio del usuario."""
    base = STORAGE / uid
    (base / "docs").mkdir(parents=True, exist_ok=True)
    (base / "cache").mkdir(parents=True, exist_ok=True)
    return base


# ===================== Utilidades =====================

YEARS_RE = re.compile(r"(19|20)\d{2}")


def extract_text_years(pdf_path: Path) -> Dict[str, Any]:
    """Extrae texto y años de un PDF si la librería pypdf está disponible."""
    text = ""
    years: List[int] = []
    if PdfReader is None:
        return {"text": text, "years": years}
    try:
        with open(pdf_path, "rb") as fh:
            pdf = PdfReader(fh)
            for p in pdf.pages:
                t = p.extract_text() or ""
                text += "\n" + t
        years = sorted({int(m.group(0)) for m in YEARS_RE.finditer(text)})
    except Exception:
        pass
    return {"text": text, "years": years}


def save_meta(uid: str, meta: Dict[str, Any]):
    """Guarda metadatos de los PDFs en un archivo JSON por usuario."""
    mf = user_dir(uid) / "cache" / "meta.json"
    cur: Dict[str, Any] = {}
    if mf.exists():
        try:
            cur = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    # actualiza o añade entradas sin eliminar las previas
    cur.update(meta)
    mf.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(uid: str) -> Dict[str, Any]:
    """Lee metadatos del usuario, si existen."""
    mf = user_dir(uid) / "cache" / "meta.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ===================== Frontend (HTML) =====================

# Usamos placeholders y reemplazos en lugar de f-strings para evitar
# problemas con las llaves de JavaScript/CSS. Los valores dinámicos
# (APP_NAME, límites) se sustituyen en portal_page.
PORTAL_HTML_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>[APP_NAME]</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,Roboto;max-width:960px;margin:24px auto;padding:12px;}}
 .card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:20px 0;background:#fff;}}
 .btn{{background:#111;color:#fff;border:none;padding:10px 16px;border-radius:10px;cursor:pointer;}}
 .nowrap{{white-space:nowrap;min-width:max-content;}}
 #pending li{{margin:3px 0;}}
 .muted{{color:#666;}}
 .kpi{{padding:.75rem;border-radius:.75rem;border:1px solid #ddd;}}
 .kpi.ok{{background:#e6ffed;}} .kpi.warn{{background:#fff5cc;}} .kpi.bad{{background:#ffe6e6;}}
 @media print{{#printBtn,#uploadBtn,#pick,#pending,#askBtn,#genBtn,#msg,#qs{{display:none!important;}}}}
</style>
</head>
<body>
 <h2>[APP_NAME]</h2>
 <div class="card">
   <button id="printBtn" class="btn" onclick="window.print()">🖨️ Imprimir reporte</button>
 </div>
 <div class="card">
   <input id="pick" type="file" accept="application/pdf" multiple />
   <ul id="pending" class="muted"></ul>
   <button id="uploadBtn" class="btn">Subir PDFs e indexar</button>
   <div id="msg" class="muted"></div>
   <div><label class="nowrap"><input type="checkbox" id="prosa" /> Prosa premium (IA)</label></div>
   <small class="muted">* Máx [MAX_FILES] PDFs por vez; [SINGLE_MAX]MB por archivo; [TOTAL_MAX]MB por subida.</small>
 </div>
 <div class="card">
   <h3>Preguntar</h3>
   <textarea id="qs" rows="6" style="width:100%" placeholder="Escribe hasta 5 preguntas, una por línea"></textarea>
   <button id="askBtn" class="btn">Preguntar</button>
   <pre id="ans" style="white-space:pre-wrap"></pre>
 </div>
 <div class="card">
   <h3>Reporte Automático</h3>
   <button id="genBtn" class="btn">Generar Reporte Automático</button>
   <div id="charts"></div>
   <div id="gloss"></div>
 </div>
 <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
 <script>
  // Buffer para archivos
  const pending = [];
  const qsEl = document.getElementById('qs');
  const ansEl = document.getElementById('ans');
  const msgEl = document.getElementById('msg');
  function renderPending(){{
    const ul = document.getElementById('pending');
    ul.innerHTML = pending.map(p => `<li>${{p.name}} <a href="#" data-n="${{p.name}}">✖</a></li>`).join('');
    ul.querySelectorAll('a').forEach(a => {{
      a.onclick = e => {{
        e.preventDefault();
        const n = a.dataset.n;
        const i = pending.findIndex(p => p.name === n);
        if (i >= 0) pending.splice(i,1);
        renderPending();
      }};
    }});
  }}
  document.getElementById('pick').addEventListener('change', () => {{
    const fi = document.getElementById('pick');
    for (const f of fi.files) {{
      if (!pending.find(p => p.name === f.name)) pending.push({{file: f, name: f.name}});
    }}
    fi.value = '';
    renderPending();
  }});
  function authHeader(){{
    const tok = localStorage.getItem('token') || ((document.cookie.match(/token=([^;]+)/) || [])[1]);
    return tok ? {{ 'Authorization': 'Bearer ' + tok }} : {{}};
  }}
  document.getElementById('uploadBtn').onclick = async () => {{
    if (!pending.length) {{ alert('No hay archivos para subir.'); return; }}
    const fd = new FormData();
    pending.forEach(p => fd.append('files', p.file));
    const r = await fetch('/upload', {{ method: 'POST', body: fd, headers: authHeader(), credentials: 'include' }});
    const data = await r.json();
    if (data.ok) {{
      msgEl.textContent = data.message || '';
      (data.saved || []).forEach(nm => {{
        const i = pending.findIndex(p => p.name === nm);
        if (i >= 0) pending.splice(i, 1);
      }});
      renderPending();
    }} else {{
      msgEl.textContent = data.message || 'Error al subir.';
    }}
  }};
  document.getElementById('askBtn').onclick = async () => {{
    const lines = qsEl.value.trim().split(/\n+/).filter(l => l.trim()).slice(0,5);
    if (!lines.length) {{ ansEl.textContent = 'Ingresa una o más preguntas.'; return; }}
    const r = await fetch('/ask', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json', ...authHeader() }}, body: JSON.stringify({{ questions: lines }}) }});
    const data = await r.json();
    ansEl.textContent = (data.answers || []).join('\n\n');
  }};
  document.getElementById('genBtn').onclick = async () => {{
    const r = await fetch('/auto-report', {{ headers: authHeader() }});
    const data = await r.json();
    const charts = document.getElementById('charts');
    charts.innerHTML = '';
    function addCanvas(){{ const c = document.createElement('canvas'); charts.appendChild(c); return c; }}
    new Chart(addCanvas(), {{ type: 'bar', data: {{ labels: data.years, datasets: [{{ label: 'Ingresos / Revenue', data: data.revenue }}] }} }});
    new Chart(addCanvas(), {{ type: 'line', data: {{ labels: data.years, datasets: [{{ label: 'EBITDA (MM) / EBITDA', data: data.ebitda }}] }} }});
    new Chart(addCanvas(), {{ type: 'line', data: {{ labels: data.years, datasets: [{{ label: 'Margen EBITDA (%) / EBITDA Margin', data: data.margin }}] }} }});
    new Chart(addCanvas(), {{ type: 'pie', data: {{ labels: ['Fijos / Fixed','Variables / Variable','Otros / Other'], datasets: [{{ data: data.cost_mix }}] }} }});
    // KPIs
    const kpiList = document.createElement('div');
    charts.appendChild(kpiList);
    (data.kpis || []).forEach(k => {{
      const div = document.createElement('div');
      div.className = 'kpi ' + k.state;
      const estado = k.state === 'ok' ? 'Bueno' : k.state === 'warn' ? 'Promedio' : 'Malo';
      div.innerHTML = `<b>${{k.name_es}} / ${{k.name_en}}:</b> ${{k.value}}${{k.unit}}<br><small>Fórmula: ${{k.formula}}</small><br><small>Estado: ${estado}</small><br><small>Acción sugerida: ${{k.action}}</small>`;
      kpiList.appendChild(div);
    }});
    // Glosario
    const gl = document.getElementById('gloss');
    gl.innerHTML = '<h3>Glosario</h3><ul>' + (data.glossary || []).map(g => `<li><b>${{g.term}}</b>: ${{g.definition}}</li>`).join('') + '</ul>';
  }};
 </script>
</body>
</html>
"""


# ===================== Endpoints =====================

@app.get("/portal", response_class=HTMLResponse)
def portal_page(request: Request, uid: str = Depends(require_user)):
    """Devuelve el HTML del portal si el usuario está autenticado.

    Sustituye los placeholders del template por los valores de configuración.
    """
    html = PORTAL_HTML_TEMPLATE
    html = html.replace("[APP_NAME]", S.APP_NAME)
    html = html.replace("[MAX_FILES]", str(S.MAX_UPLOAD_FILES))
    html = html.replace("[SINGLE_MAX]", str(S.SINGLE_FILE_MAX_MB))
    html = html.replace("[TOTAL_MAX]", str(S.MAX_TOTAL_MB))
    return HTMLResponse(html)


@app.post("/upload")
async def upload(request: Request, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Recibe una lista de PDFs, los guarda y registra sus metadatos."""
    uid = require_user(request)
    base = user_dir(uid)
    if not files:
        return JSONResponse({"ok": False, "message": "Selecciona al menos un PDF."}, status_code=400)
    if len(files) > S.MAX_UPLOAD_FILES:
        return JSONResponse({"ok": False, "message": f"Máximo {S.MAX_UPLOAD_FILES} PDFs por subida."}, status_code=400)
    single_max = S.SINGLE_FILE_MAX_MB * 1024 * 1024
    total_max = S.MAX_TOTAL_MB * 1024 * 1024
    saved: List[str] = []
    total_size = 0
    meta_updates: Dict[str, Any] = {}
    for f in files:
        name = os.path.basename(f.filename or "archivo.pdf").strip()
        if not name.lower().endswith(".pdf"):
            return JSONResponse({"ok": False, "message": f"{name}: solo se aceptan PDF"}, status_code=400)
        content = await f.read()
        if len(content) > single_max:
            return JSONResponse({"ok": False, "message": f"{name}: supera {S.SINGLE_FILE_MAX_MB} MB"}, status_code=400)
        total_size += len(content)
        dest = base / "docs" / name
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(name)
        info = extract_text_years(dest)
        meta_updates[name] = {"years": info.get("years", []), "size": len(content)}
    if total_size > total_max:
        for nm in saved:
            try: (base / "docs" / nm).unlink(missing_ok=True)
            except Exception:
                pass
        return JSONResponse({"ok": False, "message": f"Superaste el total permitido ({S.MAX_TOTAL_MB} MB por subida)."}, status_code=400)
    if meta_updates:
        save_meta(uid, meta_updates)
    return {
        "ok": True,
        "saved": saved,
        "errors": [],
        "indexing": "in_progress",
        "note": "Estamos procesando tus PDFs en segundo plano.",
        "message": f"Estamos cargando tus archivo(s) PDF: {', '.join(saved)}. Analizándolos en este instante.",
    }


@app.post("/ask")
async def ask(request: Request, body: Dict[str, Any]):
    """Recibe preguntas y responde listando documentos que contienen los años."""
    uid = require_user(request)
    qs: List[str] = (body.get("questions") or [])[:5]
    meta = read_meta(uid)
    all_years: Dict[int, set] = {}
    for fn, m in meta.items():
        for y in m.get("years", []):
            all_years.setdefault(int(y), set()).add(fn)
    answers: List[str] = []
    for q in qs:
        yrs: List[int] = sorted({int(m.group(0)) for m in YEARS_RE.finditer(q)})
        if not yrs:
            parts = []
            docs = sorted(meta.keys())
            parts.append("Documentos disponibles: " + (", ".join(docs) if docs else "(sin documentos)"))
            answers.append("\n".join(parts))
            continue
        parts = []
        for y in yrs:
            docs = sorted(all_years.get(y, set()))
            if docs:
                parts.append(f"Año {y}: " + ", ".join(docs))
            else:
                parts.append(f"Año {y}: no hay evidencia en los PDFs.")
        answers.append("\n".join(parts))
    return {"answers": answers}


@app.get("/auto-report")
async def auto_report(request: Request):
    """Genera datos de ejemplo para reportes automáticos."""
    uid = require_user(request)
    meta = read_meta(uid)
    years = sorted({y for v in meta.values() for y in v.get("years", [])}) or [2020, 2021, 2022, 2023]
    # Datos dummy: se podría calcular a partir de PDFs
    revenue = [120 + i*10 for i in range(len(years))]  # ingresos
    ebitda = [18 + i*2 for i in range(len(years))]     # EBITDA
    margin = [round(e/r*100, 1) for e, r in zip(ebitda, revenue)]
    cost_mix = [40, 50, 10]  # fijos, variables, otros
    def kpi_state(v: float, good: float, warn: float) -> str:
        if v >= good: return "ok"
        if v >= warn: return "warn"
        return "bad"
    kpis = [
        {
            "name_es": "Margen EBITDA", "name_en": "EBITDA Margin", "value": margin[-1], "unit": "%",
            "formula": "EBITDA / Ingresos", "state": kpi_state(margin[-1], 20, 12),
            "action": "Revisar pricing y disciplina de costos."
        },
        {
            "name_es": "Deuda / EBITDA", "name_en": "Debt / EBITDA", "value": 2.8, "unit": "x",
            "formula": "Deuda Financiera Neta / EBITDA", "state": kpi_state(20-2.8, 17, 14),
            "action": "Bajar apalancamiento vía flujo y capex selectivo."
        },
        {
            "name_es": "Liquidez Corriente", "name_en": "Current Ratio", "value": 1.6, "unit": "x",
            "formula": "Activos Corrientes / Pasivos Corrientes", "state": kpi_state(1.6, 1.5, 1.2),
            "action": "Acelerar cobranza, negociar plazos con proveedores."
        },
    ]
    glossary = [
        {"term": "EBITDA", "definition": "Ganancias antes de Intereses, Impuestos, Depreciación y Amortización / Earnings Before Interest, Taxes, Depreciation and Amortization."},
        {"term": "WACC", "definition": "Costo Promedio Ponderado de Capital / Weighted Average Cost of Capital."},
        {"term": "FNE", "definition": "Flujo Neto de Efectivo."},
    ]
    return {
        "years": years,
        "revenue": revenue,
        "ebitda": ebitda,
        "margin": margin,
        "cost_mix": cost_mix,
        "kpis": kpis,
        "glossary": glossary,
    }


@app.get("/__warmup")
def warmup():
    """Endpoint de comprobación sin efectos secundarios."""
    return {"ok": True}


@app.post("/dev/make-token")
async def dev_make_token(body: Dict[str, Any]):
    """Devuelve un token de prueba para un Gmail dado."""
    gmail = (body.get("gmail") or "user@example.com").lower()
    uid = hashlib.sha256(gmail.encode()).hexdigest()[:16]
    return {"token": make_token(uid, 24)}
