# Use Python 3.10 slim (Bookworm for better PyTorch compatibility)
FROM python:3.10-slim-bookworm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    HF_MODEL_URL=https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# Copy project files
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY results/ /app/results/
COPY deployment/ /app/deployment/

# Download model during build
RUN mkdir -p /app/checkpoints && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/kanish33/resnet50/resolve/main/best_model.pt', '/app/checkpoints/best_model.pt'); print('Model downloaded successfully.')"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start FastAPI application
CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
