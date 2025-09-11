import os
import re
import json
import textwrap
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from jose import jwt

try:
    # pypdf is used to extract simple text from PDFs and detect years. It is lightweight
    # compared to heavier PDF parsing libraries. If unavailable, year extraction will be skipped.
    from pypdf import PdfReader  # type: ignore[attr-defined]
except Exception:
    PdfReader = None  # type: ignore[assignment]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or defaults."""
    JWT_SECRET: str = os.getenv("JWT_SECRET", "secret-key-change-me")
    BASE_URL: str = os.getenv("BASE_URL", "")
    UPLOAD_LIMIT: int = 20 * 1024 * 1024  # 20 MB per file
    TOTAL_LIMIT: int = 100 * 1024 * 1024  # 100 MB per request
    STORAGE_DIR: str = "storage"

    class Config:
        extra = "ignore"


settings = Settings()

# Create the FastAPI application and configure CORS to accept requests from any origin.  
# If you serve this behind a trusted domain you can tighten these settings.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_dirs(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def get_user_dir(uid: str) -> str:
    """Return the base directory for a given user and ensure required subdirectories exist."""
    base = os.path.join(settings.STORAGE_DIR, uid)
    ensure_dirs(os.path.join(base, "docs"))
    ensure_dirs(os.path.join(base, "cache"))
    return base


def require_user(request: Request) -> str:
    """Extract and validate the user identifier from the request via Bearer token or cookie.

    This helper enforces that every protected endpoint has a valid JWT.  If a token is
    missing, invalid or expired, an HTTP 401 response is raised.
    """
    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="No autorizado")
    try:
        data = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        uid = data.get("uid")
        exp = data.get("exp")
        # Optional expiration check
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            raise Exception("Token expirado")
        if not uid:
            raise Exception("Token sin UID")
        return str(uid)
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido")


def extract_text_years(file_path: str) -> List[int]:
    """Extract four-digit years from a PDF file using pypdf.

    Returns a list of years found in the document. If pypdf is unavailable or
    parsing fails, an empty list is returned.
    """
    years: set[int] = set()
    if PdfReader is None:
        return []
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text() or ""
            for match in re.findall(r"\b(19|20)\d{2}\b", text):
                try:
                    y = int(match)
                    if 1900 <= y <= 2100:
                        years.add(y)
                except Exception:
                    continue
    except Exception:
        # Ignore errors from malformed or encrypted PDFs
        return []
    return sorted(years)


def save_meta(base_dir: str, doc_name: str, years: List[int]) -> None:
    """Persist a mapping of years to document names for a user.

    The meta file is stored as JSON in storage/<uid>/cache/meta.json with keys
    representing years and values as lists of document names containing that year.
    """
    meta_path = os.path.join(base_dir, "cache", "meta.json")
    data: dict[str, List[str]] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    for year in years:
        key = str(year)
        data.setdefault(key, [])
        if doc_name not in data[key]:
            data[key].append(doc_name)
    with open(meta_path, "w") as f:
        json.dump(data, f)


def read_meta(base_dir: str) -> dict[str, List[str]]:
    """Load the year-to-document mapping for a user."""
    meta_path = os.path.join(base_dir, "cache", "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def parse_years(question: str) -> List[int]:
    """Extract years or ranges from a question string.

    If multiple distinct years are found, they are treated as a range from min to max.
    Otherwise each distinct four-digit year is returned individually.
    """
    nums = [int(x) for x in re.findall(r"\b(19|20)\d{2}\b", question)]
    if not nums:
        return []
    if len(nums) >= 2:
        start, end = min(nums), max(nums)
        return list(range(start, end + 1))
    return nums


class QueryRequest(BaseModel):
    questions: List[str]

    # validate that we have between 1 and 5 questions
    @field_validator('questions')
    @classmethod
    def check_questions(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("Se requieren preguntas")
        if len(v) > 5:
            raise ValueError("Máximo 5 preguntas")
        return v


@app.get("/portal", response_class=HTMLResponse)
async def portal(request: Request) -> HTMLResponse:
    """Serve the main portal HTML page for authenticated users."""
    require_user(request)
    html = textwrap.dedent(
        """
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Portal de usuario (pase activo)</title>
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          <style>
            body { font-family: Arial, sans-serif; padding: 1rem; }
            .nowrap { white-space: nowrap; min-width: max-content; }
            #filelist { margin-top: 0.5rem; }
            #filelist li { list-style: none; margin-bottom: 0.25rem; }
            #filelist a { color: red; margin-left: 0.5rem; cursor: pointer; }
            #msg { margin: 0.5rem 0; color: green; }
            #dash { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); margin-top:1rem; }
            #cards { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 1rem; }
            .card { padding: 0.75rem; border:1px solid #ccc; border-radius:0.5rem; }
            .ok { background:#e6ffed; }
            .warn { background:#fff8e5; }
            .bad { background:#ffe6e6; }
            @media print { #controls { display:none; } }
          </style>
        </head>
        <body>
          <div id="controls">
            <h1>Portal de usuario (pase activo)</h1>
            <button id="printBtn">🖨️ Imprimir reporte</button>
            <div>
              <input id="filePicker" type="file" accept="application/pdf" multiple />
              <ul id="filelist"></ul>
              <button id="uploadBtn">Subir PDFs e indexar</button>
              <div id="msg"></div>
            </div>
            <div>
              <textarea id="questions" rows="5" placeholder="Escribe hasta 5 preguntas, una por línea"></textarea>
              <button id="askBtn">Preguntar</button>
            </div>
            <div>
              <label class="nowrap"><input type="checkbox" id="premiumChk" /> Prosa premium (IA)</label>
            </div>
            <div>
              <button id="genBtn">Generar Reporte Automático</button>
            </div>
          </div>
          <div id="answer"></div>
          <div id="dash"></div>
          <div id="cards"></div>
          <div id="gloss"></div>
        <script>
          let pending = [];
          const filePicker = document.getElementById('filePicker');
          const filelist = document.getElementById('filelist');
          const msgDiv = document.getElementById('msg');
          filePicker.addEventListener('change', () => {
            for (const f of filePicker.files) {
              if (!pending.some(p => p.name === f.name)) {
                pending.push(f);
              }
            }
            renderList();
            filePicker.value = '';
          });
          function renderList() {
            filelist.innerHTML = pending.map(p => '<li>' + p.name + ' <a data-name="' + p.name + '">✖</a></li>').join('');
            document.querySelectorAll('#filelist a').forEach(a => {
              a.onclick = () => {
                const name = a.getAttribute('data-name');
                pending = pending.filter(p => p.name !== name);
                renderList();
              };
            });
          }
          document.getElementById('uploadBtn').onclick = async () => {
            if (!pending.length) {
              alert('No hay archivos para subir');
              return;
            }
            const fd = new FormData();
            pending.forEach(f => fd.append('files', f));
            const token = localStorage.getItem('token') || '';
            const r = await fetch('/upload', {
              method: 'POST',
              headers: token ? { 'Authorization': 'Bearer ' + token } : {},
              body: fd,
              credentials: 'include'
            });
            const data = await r.json();
            if (data.message) {
              msgDiv.textContent = data.message;
            }
            if (data.saved) {
              data.saved.forEach(name => {
                pending = pending.filter(p => p.name !== name);
              });
              renderList();
            }
          };
          document.getElementById('askBtn').onclick = async () => {
            const qs = document.getElementById('questions').value.trim().split(/\n+/).filter(x => x).slice(0, 5);
            if (!qs.length) { alert('Ingresa al menos una pregunta'); return; }
            const token = localStorage.getItem('token') || '';
            const r = await fetch('/ask', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                ...(token ? { 'Authorization': 'Bearer ' + token } : {})
              },
              body: JSON.stringify({ questions: qs }),
              credentials: 'include'
            });
            const data = await r.json();
            const ans = data.answers || [];
            let html = '';
            ans.forEach((res, idx) => {
              html += '<h3>Pregunta ' + (idx + 1) + ': ' + qs[idx] + '</h3>';
              if (Array.isArray(res)) {
                res.forEach(item => {
                  if (item.year) {
                    html += '<p><strong>Año ' + item.year + ':</strong> ' + item.summary + '</p>';
                  } else {
                    html += '<p>' + item.summary + '</p>';
                  }
                });
              } else {
                html += '<p>' + res + '</p>';
              }
            });
            document.getElementById('answer').innerHTML = html;
          };
          document.getElementById('genBtn').onclick = async () => {
            const token = localStorage.getItem('token') || '';
            const r = await fetch('/auto-report', {
              method: 'GET',
              headers: token ? { 'Authorization': 'Bearer ' + token } : {},
              credentials: 'include'
            });
            const data = await r.json();
            document.getElementById('dash').innerHTML = '';
            const charts = data.charts || [];
            charts.forEach((c, idx) => {
              const canvas = document.createElement('canvas');
              canvas.id = 'chart' + idx;
              document.getElementById('dash').appendChild(canvas);
              new Chart(canvas.getContext('2d'), c.config);
            });
            const cardsDiv = document.getElementById('cards');
            cardsDiv.innerHTML = '';
            (data.kpis || []).forEach(k => {
              const div = document.createElement('div');
              let cls = '';
              if (k.state === 'good') cls = 'ok';
              else if (k.state === 'average') cls = 'warn';
              else cls = 'bad';
              div.className = 'card ' + cls;
              div.innerHTML = '<strong>' + k.name_es + ' / ' + k.name_en + '</strong><br>' +
                'Valor: ' + k.value + ' ' + (k.unit || '') + '<br>' +
                'Fórmula: ' + k.formula + '<br>' +
                'Evaluación: ' + k.state.charAt(0).toUpperCase() + k.state.slice(1) + '<br>' +
                'Acción sugerida: ' + k.action;
              cardsDiv.appendChild(div);
            });
            const glossDiv = document.getElementById('gloss');
            glossDiv.innerHTML = '';
            const gloss = data.glossary || {};
            if (Object.keys(gloss).length > 0) {
              let htmlG = '<h3>Glosario</h3><ul>';
              Object.keys(gloss).forEach(key => {
                htmlG += '<li><strong>' + key + '</strong>: ' + gloss[key] + '</li>';
              });
              htmlG += '</ul>';
              glossDiv.innerHTML = htmlG;
            }
          };
          document.getElementById('printBtn').onclick = () => {
            window.print();
          };
        </script>
        </body>
        </html>
        """
    )
    return HTMLResponse(content=html)


@app.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)) -> JSONResponse:
    """Handle PDF uploads for a user.

    Files are saved incrementally into the user's doc folder and years are extracted to build
    a metadata index.  A friendly message listing the uploaded filenames is returned.
    """
    uid = require_user(request)
    base_dir = get_user_dir(uid)
    saved: List[str] = []
    total_size = 0
    for f in files:
        # read all bytes to enforce size limits; Starlette streams automatically read on first call
        contents = await f.read()
        size = len(contents)
        if size > settings.UPLOAD_LIMIT:
            continue
        total_size += size
        if total_size > settings.TOTAL_LIMIT:
            break
        filename = f.filename or "unnamed.pdf"
        doc_path = os.path.join(base_dir, "docs", filename)
        with open(doc_path, "wb") as out:
            out.write(contents)
        saved.append(filename)
        years = extract_text_years(doc_path)
        if years:
            save_meta(base_dir, filename, years)
    if not saved:
        return JSONResponse({"ok": False, "message": "No se subió ningún archivo."}, status_code=400)
    message = f"Estamos cargando tus archivo(s) PDF: {', '.join(saved)}. Analizándolos en este instante."
    return JSONResponse({"ok": True, "saved": saved, "message": message})


@app.post("/ask")
async def ask(request: Request, query: QueryRequest) -> JSONResponse:
    """Answer up to five questions based on uploaded PDFs.

    For questions containing years, the response lists the documents that include those years.  If no
    years are found, a general answer referencing all indexed documents is returned.  This endpoint
    does not perform semantic analysis but rather points to relevant documents by year.
    """
    uid = require_user(request)
    base_dir = get_user_dir(uid)
    meta = read_meta(base_dir)
    answers: List[List[dict]] = []
    for q in query.questions:
        yrs = parse_years(q)
        if yrs:
            subans: List[dict] = []
            for y in yrs:
                docs = meta.get(str(y), [])
                if docs:
                    summary = f"Se encontraron datos para el año {y} en los documentos: {', '.join(sorted(docs))}."
                else:
                    summary = f"No se encontraron documentos con información específica para el año {y}."
                subans.append({"year": y, "summary": summary})
            answers.append(subans)
        else:
            # compile list of all docs
            docs_all = sorted({d for vals in meta.values() for d in vals})
            if docs_all:
                summary = f"La respuesta general se basa en los documentos: {', '.join(docs_all)}."
            else:
                summary = "No se encontraron documentos indexados para responder esta pregunta."
            answers.append([{"summary": summary}])
    return JSONResponse({"answers": answers})


@app.get("/auto-report")
async def auto_report(request: Request) -> JSONResponse:
    """Generate a simple automatic report with charts, KPI cards and a glossary.

    This function returns example data for demonstration purposes. In a production setting you
    would derive these values from the user's documents.
    """
    require_user(request)
    # Example data for four consecutive years
    years = [2020, 2021, 2022, 2023]
    revenue = [100, 120, 130, 140]
    ebitda = [20, 24, 27, 30]
    margin = [round(e / r * 100, 2) for e, r in zip(ebitda, revenue)]
    cost_mix = [40, 35, 25]  # Representa la mezcla de costos fijos, variables y otros
    # Bar chart: Ingresos y EBITDA
    charts: List[dict] = []
    charts.append({
        "config": {
            "type": "bar",
            "data": {
                "labels": years,
                "datasets": [
                    {"label": "Ingresos / Revenue", "data": revenue},
                    {"label": "EBITDA", "data": ebitda}
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Ingresos y EBITDA"}},
                "scales": {
                    "y": {"title": {"display": True, "text": "Valor"}},
                    "x": {"title": {"display": True, "text": "Año"}}
                }
            }
        }
    })
    # Line chart: Margen EBITDA (%)
    charts.append({
        "config": {
            "type": "line",
            "data": {
                "labels": years,
                "datasets": [
                    {"label": "Margen EBITDA (%) / EBITDA Margin (%)", "data": margin}
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Margen EBITDA"}},
                "scales": {
                    "y": {"title": {"display": True, "text": "%"}, "suggestedMin": 0},
                    "x": {"title": {"display": True, "text": "Año"}}
                }
            }
        }
    })
    # Pie chart: Composición de costos
    charts.append({
        "config": {
            "type": "pie",
            "data": {
                "labels": ["Costos Fijos / Fixed Costs", "Costos Variables / Variable Costs", "Otros / Other"],
                "datasets": [
                    {"data": cost_mix}
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": "Composición de Costos / Cost Composition"}}
            }
        }
    })
    # KPI cards: compute states and suggestions
    kpis: List[dict] = []
    last_margin = margin[-1]
    state = "good" if last_margin >= 20 else ("average" if last_margin >= 15 else "bad")
    action = (
        "Mantener control de gastos." if state == "good" else
        ("Revisar precios y costos." if state == "average" else "Iniciar plan de reducción de gastos.")
    )
    kpis.append({
        "name_es": "Margen EBITDA",
        "name_en": "EBITDA Margin",
        "value": last_margin,
        "unit": "%",
        "formula": "EBITDA / Ingresos",
        "state": state,
        "action": action
    })
    # Revenue growth rate
    growth_rates: List[float] = []
    for i in range(1, len(revenue)):
        prev = revenue[i - 1]
        curr = revenue[i]
        if prev:
            growth_rates.append(round((curr - prev) / prev * 100, 2))
    last_growth = growth_rates[-1] if growth_rates else 0
    state2 = "good" if last_growth >= 10 else ("average" if last_growth >= 0 else "bad")
    action2 = (
        "Seguir estrategia de crecimiento." if state2 == "good" else
        ("Evaluar nuevos mercados." if state2 == "average" else "Analizar causas de caída y ajustar.")
    )
    kpis.append({
        "name_es": "Crecimiento de Ingresos",
        "name_en": "Revenue Growth",
        "value": last_growth,
        "unit": "%",
        "formula": "(Ingresos actual - Ingresos previo) / Ingresos previo",
        "state": state2,
        "action": action2
    })
    # Fixed cost ratio
    total_costs = sum(cost_mix)
    fixed_ratio = round(cost_mix[0] / total_costs * 100, 2) if total_costs else 0
    state3 = "good" if fixed_ratio <= 50 else ("average" if fixed_ratio <= 70 else "bad")
    action3 = (
        "Mantener estructura de costos." if state3 == "good" else
        ("Optimizar gastos fijos." if state3 == "average" else "Reducir costos fijos.")
    )
    kpis.append({
        "name_es": "Proporción de Costos Fijos",
        "name_en": "Fixed Cost Ratio",
        "value": fixed_ratio,
        "unit": "%",
        "formula": "Costos Fijos / Costos Totales",
        "state": state3,
        "action": action3
    })
    # Glossary
    glossary = {
        "EBITDA": "Ganancias antes de Intereses, Impuestos, Depreciación y Amortización / Earnings Before Interest, Taxes, Depreciation and Amortization",
        "ROI": "Retorno sobre la Inversión / Return on Investment",
        "KPI": "Indicador Clave de Desempeño / Key Performance Indicator"
    }
    return JSONResponse({"charts": charts, "kpis": kpis, "glossary": glossary})


@app.post("/mp/create-preference")
async def mp_create_preference(request: Request) -> JSONResponse:
    """Stub MercadoPago integration.

    If the coupon equals INVESTU-100, a JWT is issued granting free access. Otherwise a message
    indicates that payment integration is not implemented. This simple implementation avoids
    external dependencies while illustrating the expected behaviour.
    """
    uid = require_user(request)
    data = await request.json()
    coupon = data.get("coupon", "")
    if coupon == "INVESTU-100":
        payload = {
            "uid": uid,
            "exp": int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp())
        }
        token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
        return JSONResponse({"token": token})
    return JSONResponse({"error": "Integración de pago no implementada"}, status_code=400)


@app.get("/__warmup")
async def warmup() -> JSONResponse:
    """Warmup endpoint used by the hosting platform to start the application."""
    return JSONResponse({"ok": True})