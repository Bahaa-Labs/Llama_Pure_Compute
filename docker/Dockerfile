# ============================================================================
# Lightweight GPU Runtime
# ============================================================================

FROM nvidia/cuda:13.0.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the exact CUDA-enabled PyTorch runtime used by the project.
RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Triton runtime.
RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    triton

# Runtime dependencies only.
COPY docker/docker.requirements-runtime.txt /tmp/requirements-runtime.txt

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    -r /tmp/requirements-runtime.txt && \
    rm -f /tmp/requirements-runtime.txt

# Copy the source tree.
COPY src/ /app/src/

# Copy the already-built native CUDA extension.
# This must exist before docker build:
#   src/llama_pure_compute/_C*.so
COPY src/llama_pure_compute/_C*.so \
    /app/src/llama_pure_compute/

# Run as non-root.
RUN useradd \
        --create-home \
        --uid 10001 \
        appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python3", "-c", "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"]