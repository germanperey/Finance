FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# FAISS necesita OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia código
COPY . .

# Render inyecta PORT
EXPOSE 10000

# Arranque: usa el PORT de Render, 1 worker (free tier) 
CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level info"]
