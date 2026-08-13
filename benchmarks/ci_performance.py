from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from llama_pure_compute.ops import (
    _rmsswiglu_forward_pytorch,
    is_cuda_backend_available,
    rmsswiglu_forward,
)

from llama_pure_compute.triton_kernels.flash_attention import (
    flash_attention_v2,
)


def timed_cuda(
    fn: Callable[[], torch.Tensor],
    warmup: int = 20,
    repetitions: int = 40,
) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    samples: list[float] = []

    for _ in range(repetitions):
        start.record()
        fn()
        end.record()

        end.synchronize()
        samples.append(start.elapsed_time(end))

    return statistics.median(samples)


def benchmark_rms_swiglu() -> dict:
    hidden_dim = 4096
    inter_dim = 11008
    num_tokens = 1
    dtype = torch.float16

    device = torch.device("cuda")

    x = torch.randn(
        num_tokens,
        hidden_dim,
        device=device,
        dtype=dtype,
    )

    rms_weight = torch.ones(
        hidden_dim,
        device=device,
        dtype=dtype,
    )

    gate_weight = (
        torch.randn(
            inter_dim,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        / hidden_dim**0.5
    )

    up_weight = (
        torch.randn(
            inter_dim,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        / hidden_dim**0.5
    )

    fused_weight = torch.cat(
        [gate_weight, up_weight],
        dim=0,
    ).contiguous()

    empty = torch.empty(
        0,
        device=device,
        dtype=dtype,
    )

    reference_latency = timed_cuda(
        lambda: _rmsswiglu_forward_pytorch(
            x,
            rms_weight,
            gate_weight,
            up_weight,
            1e-5,
        )
    )

    custom_latency = timed_cuda(
        lambda: rmsswiglu_forward(
            x,
            rms_weight,
            fused_weight,
            empty,
            1e-5,
        )
    )

    return {
        "name": "rmsnorm_swiglu",
        "dtype": "float16",
        "tokens": num_tokens,
        "reference_ms": reference_latency,
        "custom_ms": custom_latency,
        "speedup": reference_latency / custom_latency,
    }


def benchmark_flash_attention() -> dict:
    batch = 1
    heads = 32
    sequence_length = 1024
    head_dim = 128
    dtype = torch.float16

    device = torch.device("cuda")

    q = torch.randn(
        batch,
        heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=dtype,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    scale = head_dim**-0.5

    custom_latency = timed_cuda(
        lambda: flash_attention_v2(
            q=q,
            k=k,
            v=v,
            causal=True,
            sm_scale=scale,
        )
    )

    reference_latency = timed_cuda(
        lambda: F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=scale,
        )
    )

    return {
        "name": "flash_attention_v2",
        "dtype": "float16",
        "sequence_length": sequence_length,
        "reference_ms": reference_latency,
        "custom_ms": custom_latency,
        "speedup": reference_latency / custom_latency,
    }


def collect_results() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if not is_cuda_backend_available():
        raise RuntimeError(
            "Custom CUDA backend is unavailable."
        )

    properties = torch.cuda.get_device_properties(0)

    results = {
        "schema_version": 1,
        "timestamp": int(time.time()),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
        },
        "benchmarks": [
            benchmark_rms_swiglu(),
            benchmark_flash_attention(),
        ],
    }

    # Avoid importing non-serializable CUDA objects.
    _ = properties

    return results


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/performance.json"),
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "benchmarks/baselines/rtx3080.json"
        ),
    )

    parser.add_argument(
        "--update-baseline",
        action="store_true",
    )

    args = parser.parse_args()

    results = collect_results()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.update_baseline:
        args.baseline.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.baseline.write_text(
            json.dumps(
                results,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"Updated baseline: {args.baseline}"
        )
        return 0

    if not args.baseline.is_file():
        raise RuntimeError(
            f"Baseline does not exist: {args.baseline}"
        )

    baseline = json.loads(
        args.baseline.read_text(
            encoding="utf-8"
        )
    )

    current_gpu = results["environment"]["gpu"]
    baseline_gpu = baseline["environment"]["gpu"]

    current_cc = results["environment"][
        "compute_capability"
    ]
    baseline_cc = baseline["environment"][
        "compute_capability"
    ]

    if current_gpu != baseline_gpu:
        raise RuntimeError(
            "Performance baseline GPU mismatch: "
            f"{current_gpu!r} != {baseline_gpu!r}"
        )

    if current_cc != baseline_cc:
        raise RuntimeError(
            "Performance baseline compute capability mismatch: "
            f"{current_cc!r} != {baseline_cc!r}"
        )

    baseline_map = {
        item["name"]: item
        for item in baseline["benchmarks"]
    }

    failures: list[str] = []

    # 10% latency regression tolerance.
    max_latency_regression = 1.10

    # 8% minimum speedup degradation tolerance.
    min_speedup_factor = 0.92

    for current in results["benchmarks"]:
        reference = baseline_map[current["name"]]

        max_allowed_latency = (
            reference["custom_ms"]
            * max_latency_regression
        )

        min_allowed_speedup = (
            reference["speedup"]
            * min_speedup_factor
        )

        print()
        print(f"Benchmark: {current['name']}")
        print(
            f"  baseline latency: "
            f"{reference['custom_ms']:.4f} ms"
        )
        print(
            f"  current latency:  "
            f"{current['custom_ms']:.4f} ms"
        )
        print(
            f"  maximum allowed:  "
            f"{max_allowed_latency:.4f} ms"
        )
        print(
            f"  baseline speedup: "
            f"{reference['speedup']:.3f}x"
        )
        print(
            f"  current speedup:  "
            f"{current['speedup']:.3f}x"
        )
        print(
            f"  minimum allowed:   "
            f"{min_allowed_speedup:.3f}x"
        )

        if current["custom_ms"] > max_allowed_latency:
            failures.append(
                f"{current['name']}: latency regression"
            )

        if current["speedup"] < min_allowed_speedup:
            failures.append(
                f"{current['name']}: speedup regression"
            )

    if failures:
        print()
        print("PERFORMANCE GATE: FAILED")

        for failure in failures:
            print(f"  - {failure}")

        return 1

    print()
    print("PERFORMANCE GATE: PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())