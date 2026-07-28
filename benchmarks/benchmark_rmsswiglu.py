"""
Hardware-Level Performance & Bandwidth Benchmark

Executes comprehensive performance profiling for the fused RMSNorm + SwiGLU 
CUDA kernel in Llama_Pure_Compute compared to PyTorch standard baselines.

Target Hardware Focus:
    - GPU: NVIDIA GeForce RTX 3080 (Ampere GA102, 10GB VRAM, 760 GB/s HBM BW)
    - Host: Fedora Workstation (Core i5 12400F)
    
Optimization Note:
    In production LLM inference engines (e.g., vLLM, TensorRT-LLM), weight matrices
    W_gate and W_up are pre-concatenated into W_fused = [W_gate; W_up] at model load time.
    This benchmark profiles zero-copy execution using pre-fused weights.
"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

# Attempt importing custom kernel dispatcher
try:
    from llama_pure_compute.ops import rmsswiglu_forward, _rmsswiglu_forward_pytorch, is_cuda_backend_available
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from llama_pure_compute.ops import rmsswiglu_forward, _rmsswiglu_forward_pytorch, is_cuda_backend_available

# Configure Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_rmsswiglu")

# Hardware Constants (NVIDIA RTX 3080 10GB Ampere)
RTX_3080_PEAK_BW_GBS = 760.0        # 760 GB/s GDDR6X Bandwidth
RTX_3080_FP16_TFLOPS = 29.8         # ~29.77 Dense FP16 TFLOPS (Non-Sparse)
L2_CACHE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB L2 Cache on GA102


class L2CacheFlusher:
    """Utility class to flush GPU L2 cache between timing runs to prevent unrealistic warm-cache hits."""
    def __init__(self, size_bytes: int = L2_CACHE_SIZE_BYTES):
        self.buffer = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")

    def flush(self):
        self.buffer.zero_()


def calculate_rmsswiglu_bytes(num_tokens: int, hidden_dim: int, inter_dim: int, dtype: torch.dtype) -> int:
    """
    Calculates total global memory bytes transferred (Reads + Writes) for Fused RMSNorm + SwiGLU.
    
    Reads:
        - Input X: num_tokens * hidden_dim * element_size
        - RMS Weight: hidden_dim * element_size
        - Fused Weight: (2 * inter_dim) * hidden_dim * element_size
    Writes:
        - Output: num_tokens * inter_dim * element_size
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    
    bytes_read_x = num_tokens * hidden_dim * element_size
    bytes_read_rms_w = hidden_dim * element_size
    bytes_read_fused_w = (2 * inter_dim) * hidden_dim * element_size
    bytes_write_out = num_tokens * inter_dim * element_size
    
    return bytes_read_x + bytes_read_rms_w + bytes_read_fused_w + bytes_write_out


def calculate_rmsswiglu_flops(num_tokens: int, hidden_dim: int, inter_dim: int) -> int:
    """
    Calculates floating point operations (FLOPs) for RMSNorm + Single Concatenated GEMM + SwiGLU.
    
    1. RMSNorm: ~4 FLOPs/elem -> 4 * num_tokens * hidden_dim
    2. Concatenated GEMM: 2 * num_tokens * hidden_dim * (2 * inter_dim) = 4 * num_tokens * hidden_dim * inter_dim
    3. SwiGLU Activations: ~8 FLOPs/elem (SiLU(x) * y) -> 8 * num_tokens * inter_dim
    """
    flops_rmsnorm = 4 * num_tokens * hidden_dim
    flops_gemm = 4 * num_tokens * hidden_dim * inter_dim
    flops_swiglu = 8 * num_tokens * inter_dim
    
    return flops_rmsnorm + flops_gemm + flops_swiglu


def benchmark_op(
    fn, 
    args: tuple, 
    warmup: int = 30, 
    reps: int = 100, 
    flusher: Optional[L2CacheFlusher] = None
) -> float:
    """Measures precise average GPU kernel latency in milliseconds using CUDA Events."""
    # Warmup runs
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms = []
    for _ in range(reps):
        if flusher:
            flusher.flush()
            
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))

    # Return median latency to discard outlier system jitter
    latencies_ms.sort()
    return latencies_ms[len(latencies_ms) // 2]


def run_benchmark_suite(
    hidden_dim: int = 4096,
    inter_dim: int = 11008,
    batch_sizes: List[int] = [1, 4, 16, 64, 256, 1024, 4096],
    dtypes: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]
) -> List[Dict]:
    """Runs performance sweeps across shapes and precision data types."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run rmsswiglu hardware benchmarks.")

    if not is_cuda_backend_available():
        logger.warning("CUDA backend (_C) is not installed! Benchmarks will run PyTorch fallback only.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    logger.info(f"Starting RMSSwiGLU Benchmark Suite on: {gpu_name}")
    logger.info(f"Model Configuration: hidden_dim={hidden_dim}, intermediate_dim={inter_dim}")

    flusher = L2CacheFlusher()
    results = []

    for dtype in dtypes:
        dtype_str = str(dtype).split(".")[-1]
        logger.info(f"\n--- Benchmarking Precision: {dtype_str} ---")

        for num_tokens in batch_sizes:
            # Memory allocation
            x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
            rms_w = torch.ones(hidden_dim, device=device, dtype=dtype)
            gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype) / (hidden_dim ** 0.5)
            up_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype) / (hidden_dim ** 0.5)

            # Pre-fuse weights OUTSIDE the benchmark timing loop
            fused_w = torch.cat([gate_w, up_w], dim=0).contiguous()

            # 1. Benchmark PyTorch Native Reference
            torch_args = (x, rms_w, gate_w, up_w, 1e-5)
            ref_latency_ms = benchmark_op(_rmsswiglu_forward_pytorch, torch_args, flusher=flusher)

            # 2. Benchmark Fused Custom CUDA Kernel with Pre-fused Weights
            # Passing empty Tensor for up_w triggers zero-copy pre-fused GEMM path
            empty_tensor = torch.Tensor().to(device=device, dtype=dtype)
            fused_cuda_args = (x, rms_w, fused_w, empty_tensor, 1e-5)
            fused_latency_ms = benchmark_op(rmsswiglu_forward, fused_cuda_args, flusher=flusher)

            # Performance Metrics Calculations
            speedup = ref_latency_ms / max(fused_latency_ms, 1e-6)
            total_bytes = calculate_rmsswiglu_bytes(num_tokens, hidden_dim, inter_dim, dtype)
            total_flops = calculate_rmsswiglu_flops(num_tokens, hidden_dim, inter_dim)

            achieved_bw_gbs = (total_bytes / 1e9) / (fused_latency_ms / 1000.0)
            achieved_tflops = (total_flops / 1e12) / (fused_latency_ms / 1000.0)
            sol_bw_pct = (achieved_bw_gbs / RTX_3080_PEAK_BW_GBS) * 100.0
            tokens_per_sec = num_tokens / (fused_latency_ms / 1000.0)

            res = {
                "dtype": dtype_str,
                "num_tokens": num_tokens,
                "ref_ms": ref_latency_ms,
                "fused_ms": fused_latency_ms,
                "speedup": speedup,
                "bw_gbs": achieved_bw_gbs,
                "sol_bw_pct": sol_bw_pct,
                "tflops": achieved_tflops,
                "tokens_sec": tokens_per_sec
            }
            results.append(res)

            print(
                f"Tokens: {num_tokens:5d} | "
                f"PyTorch: {ref_latency_ms:7.3f} ms | "
                f"Fused CUDA: {fused_latency_ms:7.3f} ms | "
                f"Speedup: {speedup:5.2f}x | "
                f"BW: {achieved_bw_gbs:6.1f} GB/s ({sol_bw_pct:5.1f}% SOL) | "
                f"TFLOPS: {achieved_tflops:5.2f}"
            )

    return results


def print_ascii_table(results: List[Dict]):
    """Prints a Markdown/ASCII summary table for documentation and resumes."""
    print("\n" + "="*105)
    print(" " * 32 + "Llama_Pure_Compute: RMSSwiGLU Benchmark Results")
    print("="*105)
    print(f"| {'Dtype':<8} | {'Tokens':<8} | {'PyTorch (ms)':<13} | {'Fused CUDA (ms)':<16} | {'Speedup':<9} | {'Bandwidth':<12} | {'TFLOPS':<8} |")
    print("|" + "-"*10 + "|" + "-"*10 + "|" + "-"*15 + "|" + "-"*18 + "|" + "-"*11 + "|" + "-"*14 + "|" + "-"*10 + "|")
    
    for r in results:
        print(
            f"| {r['dtype']:<8} | {r['num_tokens']:<8d} | {r['ref_ms']:<13.3f} | {r['fused_ms']:<16.3f} | "
            f"{r['speedup']:<8.2f}x | {r['bw_gbs']:<6.1f} GB/s   | {r['tflops']:<8.2f} |"
        )
    print("="*105 + "\n")


def plot_results(results: List[Dict], output_path: str = "rmsswiglu_performance.png"):
    """Generates high-resolution performance plots using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="darkgrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping plot generation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    dtypes = list(set(r["dtype"] for r in results))
    colors = {"float16": "#1f77b4", "bfloat16": "#ff7f0e", "float32": "#2ca02c"}

    # Plot 1: Latency Speedup Comparison
    for dt in dtypes:
        dt_res = [r for r in results if r["dtype"] == dt]
        tokens = [r["num_tokens"] for r in dt_res]
        speedups = [r["speedup"] for r in dt_res]
        axes[0].plot(tokens, speedups, marker='o', linewidth=2, label=f"Speedup ({dt})", color=colors.get(dt, "blue"))

    axes[0].set_xscale('log', base=2)
    axes[0].set_xlabel("Number of Tokens (Batch Size / Sequence Length)")
    axes[0].set_ylabel("Speedup over PyTorch Native (x)")
    axes[0].set_title("RMSSwiGLU Kernel Speedup vs PyTorch Baseline")
    axes[0].axhline(1.0, linestyle="--", color="gray", alpha=0.7)
    axes[0].legend()

    # Plot 2: Achieved Memory Bandwidth
    for dt in dtypes:
        dt_res = [r for r in results if r["dtype"] == dt]
        tokens = [r["num_tokens"] for r in dt_res]
        bw = [r["bw_gbs"] for r in dt_res]
        axes[1].plot(tokens, bw, marker='s', linewidth=2, label=f"BW ({dt})", color=colors.get(dt, "blue"))

    axes[1].set_xscale('log', base=2)
    axes[1].set_xlabel("Number of Tokens (Batch Size / Sequence Length)")
    axes[1].set_ylabel("Achieved Bandwidth (GB/s)")
    axes[1].set_title("Memory Bandwidth Saturation (NVIDIA RTX 3080 Peak: 760 GB/s)")
    axes[1].axhline(RTX_3080_PEAK_BW_GBS, linestyle="--", color="red", label="RTX 3080 Hardware Limit")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Performance plots successfully generated and saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama_Pure_Compute RMSSwiGLU Performance Benchmark")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension size (e.g., LLaMA-3 8B = 4096)")
    parser.add_argument("--inter-dim", type=int, default=11008, help="Intermediate dimension size (e.g., LLaMA-3 8B = 11008)")
    parser.add_argument("--save-plot", action="store_true", help="Save graphical performance plot")
    args = parser.parse_args()

    # Run complete benchmark suite
    results_data = run_benchmark_suite(
        hidden_dim=args.hidden_dim,
        inter_dim=args.inter_dim
    )

    # Print markdown table summary
    print_ascii_table(results_data)

    # Plot graphical visualization if requested
    if args.save_plot:
        plot_results(results_data)