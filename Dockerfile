# Use Python 3.10 slim (Bookworm for better PyTorch compatibility)
FROM python:3.10-slim-bookworm

# Prevent Python from buffering stdout/stderr and set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    HF_MODEL_URL="https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt"

WORKDIR /app

# Install essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only packages
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining Python dependencies
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# Copy project files
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY results/ /app/results/
COPY deployment/ /app/deployment/

# Create checkpoints directory and download the model
RUN mkdir -p /app/checkpoints && \
    python -c "import urllib.request, shutil; \
req = urllib.request.Request( \
'https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt', \
headers={'User-Agent':'Mozilla/5.0'}); \
with urllib.request.urlopen(req) as r, open('/app/checkpoints/best_model.pt','wb') as f: \
    shutil.copyfileobj(r, f); \
print('Model downloaded successfully.')"

# Expose application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start FastAPI application
CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
