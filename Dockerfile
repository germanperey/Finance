FROM python:3.11-slim
WORKDIR /app

# FAISS necesita libgomp (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# dependencias python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia el código
COPY . .

# Render te inyecta PORT
EXPOSE 10000
CMD ["bash","-lc","gunicorn app:app --bind 0.0.0.0:${PORT}"]

