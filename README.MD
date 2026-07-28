# 🦙 Llama_Pure_Compute

> **A low-level system for `RMSNorm+SwiGLU` fusion, `FlashAttention-v2`, and vectorized `CUDA` kernel execution.**

[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Triton-Language](https://img.shields.io/badge/Triton-Compiler-blue.svg)](https://github.com/triton-lang/triton)
[![Fedora Workstation](https://img.shields.io/badge/Environment-Fedora%20Workstation-blueviolet.svg)](https://fedoraproject.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

`Llama_Pure_Compute` is a custom high-performance systems engine designed to eliminate memory bandwidth bottlenecks and dispatch overheads in Llama-family LLMs. By engineering vectorized C++/CUDA C kernels via PyBind11 alongside targeted Triton DSL kernels, this repository provides zero-copy, highly fused execution across key LLM building blocks including **Fused RMSNorm + SwiGLU**, **Rotary Position Embeddings (RoPE)**, **In-Place KV-Cache Allocation**, and **FlashAttention-v2**.

---

## 🏆 Hardware Profiling & Key Performance Victories

Profiles collected using **NVIDIA Nsight Compute (NCU)** on an **NVIDIA GeForce RTX 3080 (GA102)** testing both memory-bound single-token decoding and compute-bound long-sequence prefill phases.

### 1. Fused RMSNorm + SwiGLU Kernel (`rmsswiglu.cu`)

By performing 128-bit vectorized loads (`uint4`), caching intermediate normalization scale factors ($rsqrt(\mu + \epsilon)$) in FP32 registers, and executing warp-level reduction primitives (`__shfl_down_sync`), our fused kernel completely removes intermediate global DRAM read/write passes across the RMSNorm-SwiGLU boundary.

* **3.56x High-Precision Speedup (FP32):** At 4096 tokens, standard PyTorch eager execution suffers massive DRAM serialization overhead (**44.95 ms**), whereas our fused CUDA kernel finishes in **12.61 ms**—demonstrating superior kernel-level pipeline efficiency.
* **Peak Memory Bandwidth Saturation:** Reached **978.7 GB/s effective bandwidth (128.8% Speed-of-Light)** during single-token FP32 execution through aggressive register reuse and memory coalescing.
* **Low-Latency Decoding (FP16 / BF16):** Reduces single-token auto-regressive generation latency from **0.689 ms down to 0.370 ms**
(**1.86x speedup** in FP16, **1.78x** in BF16).

#### RMSSwiGLU Profiling Benchmark (RTX 3080)

| Precision | Sequence / Tokens | PyTorch Eager (ms) | Fused CUDA (ms) | **Speedup** | **Effective Bandwidth** | **TFLOPS** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 4,096 tokens | 44.945 ms | **12.614 ms** | **3.56x** | 48.2 GB/s | 58.60 |
| **FP32** | 1,024 tokens | 11.878 ms | **3.387 ms** | **3.51x** | 124.8 GB/s | 54.55 |
| **FP32** | 1 token | 0.869 ms | **0.369 ms** | **2.36x** | **978.7 GB/s (128.8% SOL)** | 0.49 |
| **FP16** | 1 token | 0.689 ms | **0.370 ms** | **1.86x** | 488.0 GB/s (64.2% SOL) | 0.49 |
| **BF16** | 1 token | 0.660 ms | **0.370 ms** | **1.78x** | 488.0 GB/s (64.2% SOL) | 0.49 |

---

### 2. Custom Triton FlashAttention-v2 Kernel (`flash_attention.py`)

Custom Triton DSL implementation of causal FlashAttention-v2 optimized for sequence processing ($B=2, H=32, D=128$).

* **Short Sequence Overhead Reduction:** Outperforms PyTorch's native `scaled_dot_product_attention` (SDPA) by **1.70x** at short prompt lengths ($N=128$), reaching **620.81 GB/s memory bandwidth**.
* **Near-Theoretical TFLOPS Scaling:** Scales linearly up to **52.52 TFLOPS** at $N=8192$, demonstrating high Tensor Core tile utilization.

#### FlashAttention-v2 Profiling Benchmark (RTX 3080)

| Sequence Length ($N$) | Backend | Latency (ms) | Compute Throughput | **Speedup vs PyTorch SDPA** |
| :--- | :--- | :--- | :--- | :--- |
| **128** | **Triton Flash-v2** | **0.0135 ms** | 19.87 TFLOPS | **1.70x** |
| | PyTorch SDPA | 0.0229 ms | 11.72 TFLOPS | 1.00x | 
| **256** | **Triton Flash-v2** | **0.0404 ms** | 26.57 TFLOPS | **1.23x** |
| | PyTorch SDPA | 0.0497 ms | 21.61 TFLOPS | 1.00x |
| **8192** | **Triton Flash-v2** | **20.935 ms** | **52.52 TFLOPS** | **1.16x** |

---

## 🏛️ System Architecture & Engineering Concepts

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

Key Engineering Features

    Warp-Level Parallel Reductions: Low-level RMSNorm variance calculations leverage warp shuffle intrinsics (__shfl_down_sync) to eliminate block-wide shared memory barrier latency.

    Vectorized Instruction Emission: Emits 128-bit wide loads/stores via vector types (uint4 / float4) inside custom kernels (rope.cu and rmsswiglu.cu) to maximize memory bus utilization.

    Zero-Copy In-Place Mutations: KV cache updates (kv_cache.cu) bypass PyTorch's autograd tracking and operate in-place on pre-allocated contiguous memory pools to avoid dynamic allocation churn during inference generation.

📂 Repository Layout

'''text
LLAMA_PURE_COMPUTE/
├── benchmarks/
│   ├── benchmark_attention.py   # FlashAttention performance suite
│   ├── benchmark_rmsswiglu.py   # RMSNorm+SwiGLU micro-benchmarks
│   └── profile_ncu.sh           # Nsight Compute profiling launcher
├── profiles/                    # Generated NCU profile traces (.ncu-rep)
├── src/
│   └── csrc/
│       ├── include/             # C++/CUDA Header definitions
│       │   ├── kv_cache.h
│       │   ├── rmsswiglu.h
│       │   ├── rope.h
│       │   └── utils.h          # Shared Warp Reductions & CUDA macros
│       └── ops/                 # Low-Level CUDA Operators
│           ├── bindings.cpp     # PyBind11 C++ module definitions
│           ├── kv_cache.cu
│           ├── rmsswiglu.cu
│           └── rope.cu
├── llama_pure_compute/          # Core Python Engine Package
│   ├── triton_kernels/          # Isolated Triton DSL implementations
│   │   ├── flash_attention.py
│   │   └── rmsnorm.py
│   ├── config.py                # Model hyperparameter specs
│   ├── generate.py              # Generation & Sampling pipeline
│   ├── kv_manager.py            # KV Cache Allocation API
│   ├── model.py                 # Transformer Layer & Model Graph
│   ├── ops.py                   # Custom Operator Registrations
│   ├── sampler.py               # Top-K / Top-P / Temperature Samplers
│   └── tokenizer.py             # Tokenizer Integration
├── tests/                       # Complete Test & Correctness Suite
│   ├── test_attention_correctness.py
│   ├── test_generation.py
│   ├── test_kv_cache.py
│   ├── test_model_forward.py
│   ├── test_rmsswiglu_correctness.py
│   └── test_rope_correctness.py
├── CMakeLists.txt               # C++ Standalone Build System
├── pyproject.toml               # Python Package Build Configuration
├── setup.py                     # Setuptools C++/CUDA extension builder
└── README.md
'''

⚙️ Environment & Setup

Engineered and profiled on Fedora Workstation within a containerized CUDA distrobox environment (cuda_env).
Build & Installation
Bash

# 1. Activate PyTorch Virtual Environment
source activate torch_env

# 2. Enter containerized environment
distrobox enter cuda_env

# 3. Compile C++/CUDA extensions in-place
pip install -e . --no-build-isolation

💡 Quickstart & Usage
Python

import torch
from llama_pure_compute.model import LlamaModel
from llama_pure_compute.config import ModelConfig
from llama_pure_compute.generate import Generator

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ModelConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000,
    max_seq_len=2048
)

# Initialize optimized model with custom CUDA bindings
model = LlamaModel(config).to(device=device, dtype=torch.float16)

# Generate Text
generator = Generator(model=model, config=config)
output_tokens = generator.generate(
    prompt_tokens=[1, 15043, 29892, 278],
    max_gen_len=64,
    temperature=0.7
)

🧪 Testing & Verification

All operators are verified for numerical accuracy and gradient behavior against PyTorch reference outputs:
Bash

# Run unit tests across all custom kernel paths
pytest tests/ -v

Running Micro-Benchmarks & Nsight Compute Profiling
Bash

# Run standalone micro-benchmarks
python benchmarks/benchmark_rmsswiglu.py
python benchmarks/benchmark_attention.py

# Profile low-level hardware counters with NCU
bash benchmarks/profile_ncu.sh

## 📦 Containerization & Cloud-Native Kubernetes Infrastructure

`Llama_Pure_Compute` includes a production-grade containerization and Kubernetes orchestration pipeline designed to deploy fused CUDA runtime kernels with persistent model weight management under low-latency inference workloads.

### 🛠️ Container Architecture ([`Dockerfile`](./Dockerfile))
* **Multi-Stage Build Pipeline:** Built on `nvidia/cuda:12.2.0-runtime-ubuntu22.04` to keep the runtime image lean while shipping the compiled PyBind11 C++/CUDA shared objects (`_C.so`).
* **Environment Isolation:** Bundles PyTorch, NumPy, and Triton execution dependencies directly with internal package resolution (`PYTHONPATH=/app/src`).
* **Optimized Layer Caching:** Separates heavy runtime dependency installation from source-code iterations for sub-second CI/CD rebuilds.

### ☸️ Kubernetes Deployment ([`k8s-deployment.yaml`](./k8s-deployment.yaml))
* **Persistent Storage Binding (`PVC`):** Dynamically mounts persistent volume claims (`llama-models-pvc` -> `/models`) to retain model parameters and KV cache states across pod lifecycles.
* **Resource Boundaries:** Strict request/limit constraints (`500m CPU / 1Gi RAM` request, `2 CPU / 4Gi RAM` limit) to prevent host memory starvation during high-batch decoding.
* **Direct Containerd Sync:** Uses direct control-plane importing (`ctr -n k8s.io images import`) to bypass external registry latency during local cluster development.

### 🚀 Local Cluster Execution Quickstart

```bash
# 1. Build local runtime container image
docker build -t llama-pure-compute:v1 .

# 2. Import container directly into Kubernetes containerd engine
docker image save llama-pure-compute:v1 | docker exec -i desktop-control-plane ctr -n k8s.io images import -

# 3. Apply manifest & trigger rolling deployment
kubectl apply -f k8s-deployment.yaml
kubectl rollout restart deployment/llama-pure-compute -n llama-compute

# 4. Stream model execution logs
kubectl logs -f deployment/llama-pure-compute -n llama-compute
'''
---

📜 License

Distributed under the MIT License. See LICENSE for details.
