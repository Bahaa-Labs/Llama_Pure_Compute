from __future__ import annotations

import argparse

import torch
import uvicorn

from llama_pure_compute.model import (
    LlamaForCausalLM,
)
from llama_pure_compute.runtime import (
    LlamaInferenceEngine,
)

from .app import configure_engine


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        required=True,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
    )

    args = parser.parse_args()

    dtype = (
        torch.float16
        if args.dtype == "fp16"
        else torch.bfloat16
    )

    engine = (
        LlamaInferenceEngine.from_pretrained(
            args.model_dir,
            device="cuda",
            dtype=dtype,
        )
    )

    configure_engine(
        engine
    )

    uvicorn.run(
        "llama_pure_compute.server.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )