# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for dlib + face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip wheel

# Install CPU-only torch first so ultralytics doesn't pull CUDA wheels (~2GB heavier)
RUN pip install --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Now install the rest (ultralytics will reuse the torch we just installed)
RUN PYTHONPATH=/install/lib/python3.13/site-packages \
    pip install --prefix=/install .

# Pre-download YOLOv8 nano model so first inference is offline-capable
WORKDIR /build/models
RUN PYTHONPATH=/install/lib/python3.13/site-packages \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Pre-download OSNet x0_25 (TorchReID) weights so first body Re-ID inference
# doesn't need to fetch from network
RUN PYTHONPATH=/install/lib/python3.13/site-packages \
    python -c "import torchreid; torchreid.models.build_model('osnet_x0_25', num_classes=1000, pretrained=True)" \
    || echo "torchreid pretrained download may have failed; will retry at runtime"
WORKDIR /build

# ---------- runtime ----------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/home/app/.config/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libpng16-16 \
    libjpeg62-turbo \
    libfreetype6 \
    libwebp7 \
    libtiff6 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 1000 -m -s /bin/bash app

COPY --from=builder /install /usr/local
COPY --from=builder /build/models/yolov8n.pt /home/app/yolov8n.pt
RUN chown -R 1000:1000 /home/app

USER 1000
WORKDIR /home/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "home_watcher.main"]
