FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV YOLO_CONFIG_DIR=/app/.cache/ultralytics
ENV XDG_CACHE_HOME=/app/.cache
ENV TORCH_HOME=/app/.cache/torch
ENV EASYOCR_MODULE_PATH=/app/.cache/easyocr

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

# Pre-download default model weights during image build instead of startup.
RUN mkdir -p "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$EASYOCR_MODULE_PATH" \
    && python - <<'PY'
from ultralytics import YOLO
from torchvision import models
import easyocr

YOLO("yolo26n.pt")
models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
easyocr.Reader(["en"], gpu=False)
PY

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m appuser \
    && mkdir -p /app/.cache/ultralytics \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
