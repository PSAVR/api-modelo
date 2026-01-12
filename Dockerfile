# Dockerfile (para API y para Worker)
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DOWNLOAD_TIMEOUT=120

# libs para soundfile/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y modelos
COPY app ./app
COPY models ./models

EXPOSE 8080

# API (en compose el worker lo sobreescribe con command:)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
