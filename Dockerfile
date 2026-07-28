# STAGE 1: Build Environment
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy project files needed for compilation
COPY Makefile* CMakeLists.txt* ./
COPY src/ ./src/
COPY include/ ./include/

# Build the executable (Adjust 'make' or 'cmake' commands based on your build system)
RUN make -j$(nproc) || (cmake -B build && cmake --build build --config Release)

# STAGE 2: Production Runtime
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS final

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app/src"

# Install Python & pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch, NumPy, and Triton
RUN pip3 install --no-cache-dir numpy triton
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ /app/src/

CMD ["python3", "-m", "llama_pure_compute.model"]