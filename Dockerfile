# ─────────────────────────────────────────────
#  Support Triage OpenEnv — Dockerfile
#  Compatible with Hugging Face Spaces (port 7860)
# ─────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py data.py graders.py environment.py server.py openenv.yaml ./

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]