<div align="center">

# 🦙 Llama_Pure_Compute

### Production-Ready CUDA & Triton Runtime for High-Performance Llama Inference

High-performance compute runtime for Llama-family models featuring **custom CUDA kernels**, **Triton FlashAttention-v2**, **vectorized Rotary Position Embeddings**, **in-place KV cache management**, and **PyBind11 native operators** for low-latency autoregressive inference.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900?logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-DSL-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## Overview

Modern Llama inference is dominated by memory-bound operations such as **RMSNorm**, **SwiGLU**, **Rotary Position Embeddings (RoPE)**, **KV cache updates**, and **attention**. Each introduces additional kernel launches, global memory traffic, and synchronization overhead that reduce GPU utilization.

**Llama_Pure_Compute** addresses these bottlenecks by implementing optimized CUDA and Triton kernels that minimize memory movement, fuse execution stages, and integrate directly into PyTorch through native C++ extensions.

The project focuses on building reusable, production-oriented compute primitives rather than modifying model architectures.

---

## ✨ Features

- 🚀 Fused **RMSNorm + SwiGLU** CUDA kernel
- ⚡ Triton **FlashAttention-v2** implementation
- 🧠 Vectorized Rotary Position Embeddings (RoPE)
- 📦 In-place KV Cache allocation and updates
- 🔥 Native PyBind11 CUDA operators
- 📊 Nsight Compute performance profiling
- 🐳 Docker-ready deployment
- ☸️ Kubernetes deployment support
- ✅ PyTorch-native integration

---

## Table of Contents

- Performance
- System Architecture
- Repository Structure
- Installation
- Quick Start
- Benchmarking
- Testing
- Deployment
- License

---

# 📈 Performance

All benchmarks were collected on an **NVIDIA GeForce RTX 3080 (GA102)** using **NVIDIA Nsight Compute**, covering both **memory-bound single-token decoding** and **compute-bound long-context prefill** workloads. :contentReference[oaicite:1]{index=1}

## Fused RMSNorm + SwiGLU

| Metric | Result |
|---------|-------:|
| Maximum FP32 Speedup | **3.56×** |
| FP16 Decoding Speedup | **1.86×** |
| BF16 Decoding Speedup | **1.78×** |
| Peak Memory Bandwidth | **978.7 GB/s** |
| Peak Throughput | **58.60 TFLOPS** |

### Optimization Highlights

- 128-bit vectorized memory loads (`uint4`)
- Warp-level reductions (`__shfl_down_sync`)
- FP32 accumulation for numerical stability
- Register reuse to reduce global memory traffic
- Single fused kernel replacing multiple PyTorch operators

---

## Triton FlashAttention-v2

| Metric | Result |
|---------|-------:|
| Best Speedup | **1.70×** |
| Peak Compute | **52.52 TFLOPS** |
| Peak Memory Bandwidth | **620.81 GB/s** |

Designed for efficient causal attention using Triton DSL with optimized Tensor Core utilization across both short and long sequence lengths. :contentReference[oaicite:2]{index=2}

---

# 🏛️ System Architecture

Llama_Pure_Compute is organized into four independent layers, separating the Python runtime, native CUDA backend, custom operators, and benchmarking infrastructure.

```text
                               Llama_Pure_Compute
┌─────────────────────────────────────────────────────────────────────────────┐
│                  High-Performance Llama Compute Runtime                     │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        │                       │                        │
        ▼                       ▼                        ▼
┌──────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ Python Runtime   │   │ Native Backend      │   │ Profiling & Testing │
├──────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ Model            │   │ CUDA Kernels        │   │ Nsight Compute      │
│ Generation       │   │ Triton Kernels      │   │ Benchmarks          │
│ KV Manager       │   │ PyBind11            │   │ Correctness Tests   │
│ Sampling         │   │ Memory Operations   │   │ Performance Reports │
└──────────────────┘   └─────────────────────┘   └─────────────────────┘
                                │
                                ▼
                    Optimized Llama Inference Runtime
```

---

## Engineering Highlights

| Component | Purpose |
|-----------|---------|
| **Fused CUDA Kernels** | Reduce kernel launches and intermediate memory traffic |
| **Triton FlashAttention-v2** | High-throughput causal attention using Tensor Cores |
| **PyBind11 Extensions** | Native CUDA operators integrated directly into PyTorch |
| **Vectorized RoPE** | Efficient rotary position embedding computation |
| **In-place KV Cache** | Eliminate allocation overhead during autoregressive decoding |
| **Nsight Compute** | Low-level GPU profiling and performance analysis |

---

# 📂 Repository Structure

```text
+--------------------------------------------------------------------------+
|                                    Llama_Pure_Compute                                 
+--------------------------------------------------------------------------+
|  High-Level APIs (llama_pure_compute/)                                                 
|  ├── model.py           --> Full Llama Architecture & Forward Graph                   
|  ├── ops.py             --> PyTorch Custom Op Interfaces & Autograd Dispatch         
|  ├── kv_manager.py      --> In-Place KV Cache Allocator & Layout Controller           
|  ├── generate.py        --> Auto-regressive Generation Engine                         
|  └── triton_kernels/    --> Pure Python Triton DSL Implementations (FlashAttn, Norm) 
+--------------------------------------------------------------------------+
|  Custom C++/CUDA Core (src/csrc/)                                                    
|  ├── ops/                                                                           |
|  │   ├── rmsswiglu.cu   --> Fused RMSNorm + SwiGLU FP32-Accumulated Kernel            |
|  │   ├── rope.cu        --> Vectorized Rotary Positional Embeddings                   
|  │   ├── kv_cache.cu    --> Direct In-Place Cache Mutation & Strided Copy            |
|  │   └── bindings.cpp   --> PyBind11 C++ Extensions                                   
|  └── include/           --> Low-Level CUDA Headers (utils.h, rope.h,  
                                        rmsswiglu.h, kv_cache.h)           
+--------------------------------------------------------------------------+

---

## Module Overview

| Module | Responsibility |
|---------|----------------|
| **src/csrc** | Native CUDA kernels and PyBind11 bindings |
| **triton_kernels** | Triton implementations of compute-intensive operators |
| **model.py** | Llama model architecture and forward execution |
| **generate.py** | Autoregressive generation pipeline |
| **kv_manager.py** | KV cache allocation and update logic |
| **ops.py** | Python interface to native CUDA operators |
| **benchmarks/** | Micro-benchmarks and Nsight Compute profiling |
| **tests/** | Numerical correctness and regression testing |

---

# ⚙️ Kernel Optimizations

The runtime focuses on reducing memory traffic, minimizing synchronization overhead, and maximizing Tensor Core utilization.

### Fused RMSNorm + SwiGLU

- 128-bit vectorized memory loads (`uint4`)
- FP32 accumulation for numerical stability
- Warp-level reductions using `__shfl_down_sync`
- Register reuse to reduce DRAM accesses
- Single fused kernel replacing multiple PyTorch operators

---

### Triton FlashAttention-v2

- Causal attention implementation
- Block-wise softmax computation
- Optimized Tensor Core utilization
- Reduced memory movement for long-context inference

---

### Vectorized RoPE

- Fused sine/cosine computation
- Vectorized memory access
- Reduced instruction count
- Optimized for autoregressive decoding

---

### In-place KV Cache

- Zero-copy cache updates
- Pre-allocated contiguous memory pools
- Elimination of repeated allocations
- Optimized for low-latency token generation

---

# 🚀 Installation

## Prerequisites

- Python **3.10+**
- CUDA **12.0+**
- PyTorch **2.x**
- NVIDIA GPU with Compute Capability **7.5+**
- CMake **3.20+**

Clone the repository and install the project.

```bash
git clone https://github.com/<your-username>/Llama_Pure_Compute.git

cd Llama_Pure_Compute

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install --upgrade pip

pip install -e .
```

---

# ⚡ Quick Start

## Build Native CUDA Extensions

```bash
python setup.py build_ext --inplace
```

---

## Run Inference

```bash
python examples/generate.py \
    --model llama-7b \
    --prompt "Explain FlashAttention."
```

---

## Execute Benchmarks

CUDA Kernels

```bash
python benchmarks/benchmark_rmsswiglu.py
```

FlashAttention

```bash
python benchmarks/benchmark_attention.py
```

Nsight Compute

```bash
bash benchmarks/profile_ncu.sh
```

---

# 🧪 Testing

Run the complete correctness suite.

```bash
pytest -v
```

Run a single module.

```bash
pytest tests/test_rope.py
```

Continuous Integration validates:

- CUDA extension compilation
- Kernel correctness
- Numerical accuracy
- Performance regression tests
- Static analysis
- Code formatting

---

# 🐳 Deployment

Build the container image.

```bash
docker build -t llama-pure-compute .
```

Run the container.

```bash
docker run --gpus all \
    -it \
    llama-pure-compute
```

For production deployments, Kubernetes manifests and deployment configurations are available in the project.

---

# 🛣️ Roadmap

## Current

- ✅ Fused RMSNorm + SwiGLU CUDA Kernel
- ✅ Triton FlashAttention-v2
- ✅ Vectorized RoPE
- ✅ In-place KV Cache
- ✅ PyBind11 Runtime
- ✅ Nsight Compute Profiling
- ✅ Benchmark Suite
- ✅ Docker Support

---

## Planned

- ⏳ FlashAttention-3
- ⏳ FP8 Inference
- ⏳ Multi-GPU Tensor Parallelism
- ⏳ CUDA Graph Capture
- ⏳ Quantized KV Cache
- ⏳ Continuous Batching
- ⏳ PagedAttention
- ⏳ Multi-Node Inference

---

# 🤝 Contributing

Contributions are welcome.

Areas of interest include:

- CUDA kernel optimization
- Triton kernels
- Llama inference optimizations
- Memory management
- Benchmarking
- Documentation
- Testing

Before opening a Pull Request, please ensure:

- Code is formatted
- Tests pass
- Benchmarks remain reproducible
- Documentation is updated where applicable

---

# 📚 References

This project builds upon ideas and research from the following technologies:

- PyTorch
- CUDA
- Triton
- FlashAttention
- PyBind11
- Hugging Face Transformers
- Nsight Compute

---

# 📄 License

Released under the **MIT License**.

---

<div align="center">

## ⭐ Support the Project

If you find **Llama_Pure_Compute** useful, consider giving the repository a **Star**.

It helps improve project visibility, supports future development, and encourages continued open-source contributions.

---

**Built for high-performance GPU compute, efficient Llama inference, and production-grade systems engineering.**

</div>
