"""
Benchmarks Triton FlashAttention-v2 against PyTorch's native SDPA 
(Flash-Attention / cuDNN backends) across sequence lengths and precisions.
Measures latency (ms), achieved TFLOPS, and memory throughput (GB/s).
"""
import torch
import triton
from llama_pure_compute.triton_kernels.flash_attention import flash_attention_v2

def compute_attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool = True
) -> float:
    """
    Computes total floating-point operations for attention forward pass.
    
    1. Q * K^T: 2 * B * H * S * S * D
    2. Softmax * V: 2 * B * H * S * S * D
    Total Dense FLOPs = 4 * B * H * S^2 * D
    Causal mask reduces FLOPs by half (~2 * B * H * S^2 * D).
    """
    total_flops = 4.0 * batch_size * num_heads * (seq_len ** 2) * head_dim
    if causal:
        total_flops /= 2.0
    return total_flops


def compute_attention_io_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_bytes: int = 2
) -> float:
    """
    Computes total memory I/O bytes transferred for FlashAttention (HBM <-> SRAM).
    FlashAttention reads Q, K, V and writes Out once from HBM:
    Total Bytes = (Q + K + V + Out) = 4 * B * H * S * D * dtype_size
    """
    return 4.0 * batch_size * num_heads * seq_len * head_dim * dtype_bytes


def benchmark_attention(
    batch_size: int = 2,
    num_heads: int = 32,
    head_dim: int = 128,
    causal: bool = True,
    dtype: torch.dtype = torch.float16,
):
    if not torch.cuda.is_available():
        print("CUDA device is required to run FlashAttention benchmarks.")
        return

    device = "cuda"
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    dtype_bytes = 2

    # Sequence lengths to profile
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    print("=" * 110)
    print(f" FlashAttention-v2 vs PyTorch SDPA Benchmark")
    print(f" Config: Batch Size={batch_size}, Heads={num_heads}, Head Dim={head_dim}, "
          f"Causal={causal}, Dtype={dtype_str}")
    print(f" Device: {torch.cuda.get_device_name(0)}")
    print("=" * 110)
    print(f"{'Seq Len':<10} | {'Backend':<18} | {'Latency (ms)':<14} | {'TFLOPS':<12} | {'Bandwidth (GB/s)':<18} | {'Speedup':<10}")
    print("-" * 110)

    for seq_len in seq_lengths:
        sm_scale = 1.0 / (head_dim ** 0.5)

        # Allocate input tensors
        q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
        k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
        v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)

        # 1. Benchmark Triton FlashAttention-v2
        fn_triton = lambda: flash_attention_v2(q, k, v, causal=causal, sm_scale=sm_scale)
        ms_triton = triton.testing.do_bench(fn_triton, warmup=25, rep=100)

        # 2. Benchmark PyTorch SDPA
        fn_sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale
        )
        ms_sdpa = triton.testing.do_bench(fn_sdpa, warmup=25, rep=100)

        # Compute Metrics
        flops = compute_attention_flops(batch_size, num_heads, seq_len, head_dim, causal=causal)
        io_bytes = compute_attention_io_bytes(batch_size, num_heads, seq_len, head_dim, dtype_bytes=dtype_bytes)

        tflops_triton = (flops / (ms_triton * 1e-3)) / 1e12
        bandwidth_triton = (io_bytes / (ms_triton * 1e-3)) / 1e9

        tflops_sdpa = (flops / (ms_sdpa * 1e-3)) / 1e12
        bandwidth_sdpa = (io_bytes / (ms_sdpa * 1e-3)) / 1e9

        speedup = ms_sdpa / ms_triton

        print(f"{seq_len:<10} | {'Triton Flash-v2':<18} | {ms_triton:<14.4f} | {tflops_triton:<12.2f} | {bandwidth_triton:<18.2f} | {speedup:<10.2f}x")
        print(f"{'':<10} | {'PyTorch SDPA':<18} | {ms_sdpa:<14.4f} | {tflops_sdpa:<12.2f} | {bandwidth_sdpa:<18.2f} | {'1.00x':<10}")
        print("-" * 110)


if __name__ == "__main__":
    # Run FP16 Benchmark (Llama 3 layout: head_dim=128)
    benchmark_attention(batch_size=2, num_heads=32, head_dim=128, causal=True, dtype=torch.float16)