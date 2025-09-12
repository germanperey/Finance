FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# Dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

EXPOSE 10000

# Arranque: Render inyecta $PORT en runtime
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level info"]




