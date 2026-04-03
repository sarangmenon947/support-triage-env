# ─────────────────────────────────────────────
#  Support Triage OpenEnv — Dockerfile
#  Compatible with Hugging Face Spaces (port 7860)
# ─────────────────────────────────────────────

FROM python:3.11-slim

# Keeps Python output unbuffered so [STEP] logs appear immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py data.py graders.py environment.py server.py tools.py openenv.yaml ./

# HF Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]