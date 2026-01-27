FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_HOME=/tmp/hf

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache /tmp/hf

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
