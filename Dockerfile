FROM python:3.11-slim
WORKDIR /app

# instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia TODO (incluye templates/, static/, etc.)
COPY . .

# Render provee PORT
EXPOSE 10000

# FastAPI con Gunicorn + worker Uvicorn
CMD ["bash","-lc","gunicorn -k uvicorn.workers.UvicornWorker -w 1 -t 120 app:app --bind 0.0.0.0:${PORT}"]
