from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import triton

from llama_pure_compute.triton_kernels.flash_attention import (
    flash_attention_v2,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_HEADS = 32
DEFAULT_HEAD_DIM = 128
DEFAULT_CAUSAL = True
DEFAULT_DTYPE = torch.float16

SEQUENCE_LENGTHS = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)

WARMUP = 25
REPETITIONS = 100


# ============================================================================
# Result structure
# ============================================================================

@dataclass
class BenchmarkResult:
    name: str
    latency_ms: float
    tflops: float
    bandwidth_gbps: float


@dataclass
class RowResult:
    seq_len: int
    triton: BenchmarkResult
    sdpa: BenchmarkResult
    flash: BenchmarkResult | None

    @property
    def triton_vs_sdpa(self) -> float:
        return self.sdpa.latency_ms / self.triton.latency_ms

    @property
    def triton_vs_flash(self) -> float | None:
        if self.flash is None:
            return None

        return self.flash.latency_ms / self.triton.latency_ms


# ============================================================================
# Environment helpers
# ============================================================================

def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark. "
            "torch.cuda.is_available() returned False."
        )

    return torch.device("cuda")


def validate_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    print(f"PyTorch:             {torch.__version__}")
    print(f"PyTorch CUDA:        {torch.version.cuda}")
    print(f"GPU:                 {device_name}")
    print(f"Compute Capability:  {capability}")
    print()


# ============================================================================
# FLOP / bandwidth model
# ============================================================================

def attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> float:
    """
    FLOP model used consistently across all three backends.

    QK^T + AV:
        2 * B * H * S^2 * D
    """
    return 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim


def attention_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> float:
    """
    Approximate HBM traffic:

        Q + K + V + O

    This is an effective bandwidth metric rather than a hardware-counter
    measurement.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()

    elements = (
        4
        * batch_size
        * num_heads
        * seq_len
        * head_dim
    )

    return float(elements * element_size)


def metrics(
    latency_ms: float,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[float, float]:
    flops = attention_flops(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    bytes_moved = attention_bytes(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
    )

    seconds = latency_ms / 1000.0

    tflops = flops / seconds / 1e12
    bandwidth_gbps = bytes_moved / seconds / 1e9

    return tflops, bandwidth_gbps


# ============================================================================
# Benchmark wrappers
# ============================================================================

def benchmark_callable(
    fn: Callable[[], torch.Tensor],
) -> float:
    """
    Measure CUDA execution latency with Triton's benchmark harness.
    """

    latency_ms = triton.testing.do_bench(
        fn,
        warmup=WARMUP,
        rep=REPETITIONS,
    )

    return float(latency_ms)


def run_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
) -> torch.Tensor:

    return flash_attention_v2(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )


def run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
) -> torch.Tensor:

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=scale,
    )


def run_flash_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    scale: float,
) -> torch.Tensor:

    # PyTorch 2.x API.
    from torch.nn.attention import (
        SDPBackend,
        sdpa_kernel,
    )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            scale=scale,
        )


# ============================================================================
# Correctness
# ============================================================================

def verify_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    name: str,
) -> None:

    if reference.shape != candidate.shape:
        raise AssertionError(
            f"{name}: shape mismatch: "
            f"{candidate.shape} != {reference.shape}"
        )

    if reference.dtype != candidate.dtype:
        raise AssertionError(
            f"{name}: dtype mismatch: "
            f"{candidate.dtype} != {reference.dtype}"
        )

    max_abs_error = (
        torch.max(
            torch.abs(
                reference.float()
                - candidate.float()
            )
        )
        .item()
    )

    if reference.dtype == torch.float16:
        atol = 1e-2
        rtol = 1e-2
    elif reference.dtype == torch.bfloat16:
        atol = 2e-2
        rtol = 2e-2
    else:
        atol = 1e-4
        rtol = 1e-4

    torch.testing.assert_close(
        candidate,
        reference,
        atol=atol,
        rtol=rtol,
    )

    print(
        f"    {name:<28}"
        f"PASS"
        f" | max_abs_error={max_abs_error:.6f}"
    )


# ============================================================================
# Single sequence benchmark
# ============================================================================

def benchmark_sequence(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    verify: bool = True,
) -> RowResult:

    device = get_device()

    scale = head_dim ** -0.5

    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # ------------------------------------------------------------------------
    # Warmup / compilation
    # ------------------------------------------------------------------------

    out_triton = run_triton(
        q,
        k,
        v,
        causal,
        scale,
    )

    out_sdpa = run_sdpa(
        q,
        k,
        v,
        causal,
        scale,
    )

    flash_available = True
    out_flash = None

    try:
        out_flash = run_flash_backend(
            q,
            k,
            v,
            causal,
            scale,
        )
    except Exception as exc:
        flash_available = False

        print(
            f"    PyTorch FlashAttention backend unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    torch.cuda.synchronize()

    # ------------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------------

    if verify:
        print("  Correctness:")

        verify_close(
            out_sdpa,
            out_triton,
            "Triton vs SDPA",
        )

        if flash_available and out_flash is not None:
            verify_close(
                out_sdpa,
                out_flash,
                "Flash backend vs SDPA",
            )

            verify_close(
                out_flash,
                out_triton,
                "Triton vs Flash backend",
            )

    # ------------------------------------------------------------------------
    # Benchmark functions
    # ------------------------------------------------------------------------

    fn_triton = lambda: run_triton(
        q,
        k,
        v,
        causal,
        scale,
    )

    fn_sdpa = lambda: run_sdpa(
        q,
        k,
        v,
        causal,
        scale,
    )

    fn_flash = lambda: run_flash_backend(
        q,
        k,
        v,
        causal,
        scale,
    )

    # ------------------------------------------------------------------------
    # Measure
    # ------------------------------------------------------------------------

    ms_triton = benchmark_callable(fn_triton)
    ms_sdpa = benchmark_callable(fn_sdpa)

    if flash_available:
        ms_flash = benchmark_callable(fn_flash)
    else:
        ms_flash = None

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    triton_tflops, triton_bw = metrics(
        ms_triton,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype,
    )

    sdpa_tflops, sdpa_bw = metrics(
        ms_sdpa,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype,
    )

    triton_result = BenchmarkResult(
        name="Triton Flash-v2",
        latency_ms=ms_triton,
        tflops=triton_tflops,
        bandwidth_gbps=triton_bw,
    )

    sdpa_result = BenchmarkResult(
        name="PyTorch SDPA",
        latency_ms=ms_sdpa,
        tflops=sdpa_tflops,
        bandwidth_gbps=sdpa_bw,
    )

    if ms_flash is not None:
        flash_tflops, flash_bw = metrics(
            ms_flash,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            dtype,
        )

        flash_result = BenchmarkResult(
            name="PyTorch FlashAttention",
            latency_ms=ms_flash,
            tflops=flash_tflops,
            bandwidth_gbps=flash_bw,
        )
    else:
        flash_result = None

    return RowResult(
        seq_len=seq_len,
        triton=triton_result,
        sdpa=sdpa_result,
        flash=flash_result,
    )


# ============================================================================
# Output formatting
# ============================================================================

def print_header(
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:

    dtype_name = str(dtype).replace(
        "torch.",
        "",
    )

    print("=" * 118)
    print(
        " FlashAttention-v2 vs PyTorch SDPA "
        "vs PyTorch FlashAttention"
    )
    print(
        f" Config: Batch Size={batch_size}, "
        f"Heads={num_heads}, "
        f"Head Dim={head_dim}, "
        f"Causal={causal}, "
        f"Dtype={dtype_name}"
    )
    print(
        f" Device: {torch.cuda.get_device_name(0)}"
    )
    print("=" * 118)

    print(
        f"{'Seq Len':<10}"
        f"| {'Backend':<25}"
        f"| {'Latency (ms)':>14}"
        f"| {'TFLOPS':>10}"
        f"| {'Bandwidth (GB/s)':>18}"
        f"| {'Speedup vs SDPA':>17}"
        f"| {'Speedup vs Flash':>17}"
    )

    print("-" * 118)


def print_row(
    row: RowResult,
) -> None:

    speedup_sdpa = row.triton_vs_sdpa

    speedup_flash = row.triton_vs_flash

    flash_speed = (
        f"{speedup_flash:>15.2f}x"
        if speedup_flash is not None
        else f"{'N/A':>17}"
    )

    print(
        f"{row.seq_len:<10}"
        f"| {'Triton Flash-v2':<25}"
        f"| {row.triton.latency_ms:>14.4f}"
        f"| {row.triton.tflops:>10.2f}"
        f"| {row.triton.bandwidth_gbps:>18.2f}"
        f"| {speedup_sdpa:>15.2f}x"
        f"| {flash_speed}"
    )

    print(
        f"{'':<10}"
        f"| {'PyTorch SDPA':<25}"
        f"| {row.sdpa.latency_ms:>14.4f}"
        f"| {row.sdpa.tflops:>10.2f}"
        f"| {row.sdpa.bandwidth_gbps:>18.2f}"
        f"| {'1.00x':>17}"
        f"| {'-':>17}"
    )

    if row.flash is not None:
        flash_vs_sdpa = (
            row.sdpa.latency_ms
            / row.flash.latency_ms
        )

        print(
            f"{'':<10}"
            f"| {'PyTorch FlashAttention':<25}"
            f"| {row.flash.latency_ms:>14.4f}"
            f"| {row.flash.tflops:>10.2f}"
            f"| {row.flash.bandwidth_gbps:>18.2f}"
            f"| {flash_vs_sdpa:>15.2f}x"
            f"| {'1.00x':>17}"
        )

    print("-" * 118)


# ============================================================================
# Full benchmark
# ============================================================================

def benchmark_attention(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    causal: bool = DEFAULT_CAUSAL,
    dtype: torch.dtype = DEFAULT_DTYPE,
    sequence_lengths: tuple[int, ...] = SEQUENCE_LENGTHS,
) -> list[RowResult]:

    validate_environment()

    print_header(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
    )

    results: list[RowResult] = []

    for seq_len in sequence_lengths:

        print(
            f"\nRunning sequence length: {seq_len}"
        )

        result = benchmark_sequence(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            causal=causal,
            dtype=dtype,
            verify=True,
        )

        results.append(result)

        print_row(result)

    return results


# ============================================================================
# Summary
# ============================================================================

def print_summary(
    results: list[RowResult],
) -> None:

    print()
    print("=" * 118)
    print(" Summary")
    print("=" * 118)

    print(
        f"{'Seq':<8}"
        f"{'Triton/SDPA':>16}"
        f"{'Triton/Flash':>18}"
        f"{'Triton TFLOPS':>18}"
    )

    print("-" * 118)

    for result in results:

        flash_ratio = (
            f"{result.triton_vs_flash:.3f}x"
            if result.triton_vs_flash is not None
            else "N/A"
        )

        print(
            f"{result.seq_len:<8}"
            f"{result.triton_vs_sdpa:>16.3f}x"
            f"{flash_ratio:>18}"
            f"{result.triton.tflops:>18.2f}"
        )

    print("=" * 118)

    best_sdpa = max(
        results,
        key=lambda x: x.triton_vs_sdpa,
    )

    print(
        f"Best Triton vs SDPA: "
        f"{best_sdpa.triton_vs_sdpa:.3f}x "
        f"at sequence length {best_sdpa.seq_len}"
    )

    flash_results = [
        x
        for x in results
        if x.triton_vs_flash is not None
    ]

    if flash_results:
        best_flash = max(
            flash_results,
            key=lambda x: x.triton_vs_flash,
        )

        print(
            f"Best Triton vs PyTorch FlashAttention: "
            f"{best_flash.triton_vs_flash:.3f}x "
            f"at sequence length {best_flash.seq_len}"
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = benchmark_attention(
        batch_size=2,
        num_heads=32,
        head_dim=128,
        causal=True,
        dtype=torch.float16,
    )

    print_summary(results)