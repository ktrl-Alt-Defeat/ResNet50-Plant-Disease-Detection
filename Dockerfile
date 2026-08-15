# Use slim Python 3.10 image for optimal size and security
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr and set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    HF_MODEL_URL="https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt"

WORKDIR /app

# Install essential system packages (curl for healthchecks, libgomp1 for PyTorch CPU runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only packages to keep image lightweight (~700MB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install deployment dependencies
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# Copy source code, configurations, results, and deployment module
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY results/ /app/results/
COPY deployment/ /app/deployment/

# Pre-download best_model.pt from Hugging Face during image build so deployment boots instantly
RUN python -c "import urllib.request, os, shutil; os.makedirs('checkpoints', exist_ok=True); req = urllib.request.Request('https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt', headers={'User-Agent': 'Mozilla/5.0'}); res = urllib.request.urlopen(req); f = open('checkpoints/best_model.pt', 'wb'); shutil.copyfileobj(res, f); f.close(); print('[Docker Build] Downloaded best_model.pt from Hugging Face')"

# Expose HTTP port
EXPOSE 8080

# Configure container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Launch FastAPI web app using Uvicorn with dynamic $PORT binding for Render
CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
