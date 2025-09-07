FROM python:3.11-slim
WORKDIR /app

# instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia TODO (incluye templates/, static/, etc.)
COPY . .

# Render provee PORT => no lo fijes, solo expón opcionalmente
EXPOSE 10000

# si tu archivo es app.py y el objeto se llama app → app:app
CMD ["bash","-lc","gunicorn app:app --bind 0.0.0.0:${PORT}"]
