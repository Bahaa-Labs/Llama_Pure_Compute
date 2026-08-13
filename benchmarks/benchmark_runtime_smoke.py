from __future__ import annotations

import argparse

import torch

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.generate import GenerationConfig
from llama_pure_compute.model import LlamaForCausalLM
from llama_pure_compute.runtime import (
    GenerationRequest,
    LlamaInferenceEngine,
)


def build_tiny_config(
    *,
    vocab_size: int,
    max_seq_len: int,
) -> LlamaModelConfig:

    return LlamaModelConfig(
        vocab_size=vocab_size,
        dim=128,
        inter_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    config = build_tiny_config(
        vocab_size=args.vocab_size,
        max_seq_len=(
            args.prompt_length
            + args.max_new_tokens
        ),
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = (
        LlamaForCausalLM(
            config
        )
        .cuda()
        .half()
        .eval()
    )

    engine = LlamaInferenceEngine(
        model
    )

    prompt = list(
        range(
            1,
            args.prompt_length + 1,
        )
    )

    result = engine.generate(
        GenerationRequest(
            prompt_tokens=prompt,
            generation=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                repetition_penalty=1.0,
            ),
        )
    )

    metrics = result.metrics

    print()
    print("=" * 72)
    print("Llama_Pure_Compute Phase 3 Runtime Smoke Benchmark")
    print("=" * 72)
    print(
        f"GPU:                  "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Prompt tokens:        "
        f"{metrics.prompt_tokens}"
    )
    print(
        f"Generated tokens:     "
        f"{metrics.generated_tokens}"
    )
    print(
        f"TTFT:                 "
        f"{metrics.ttft_ms:.3f} ms"
    )
    print(
        f"Total latency:        "
        f"{metrics.total_latency_ms:.3f} ms"
    )
    print(
        f"ITL:                  "
        f"{metrics.itl_ms:.3f} ms"
    )
    print(
        f"Throughput:           "
        f"{metrics.tokens_per_second:.2f} tok/s"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())