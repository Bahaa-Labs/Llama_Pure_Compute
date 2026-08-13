from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from . import __version__, is_cuda_backend_available
from .generate import LlamaGenerator
from .model import LlamaForCausalLM


def _parse_prompt_tokens(value: str) -> list[int]:
    try:
        tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "prompt tokens must be comma-separated integers, e.g. 1,42,128"
        ) from exc

    if not tokens:
        raise argparse.ArgumentTypeError("at least one prompt token is required")

    if any(token < 0 for token in tokens):
        raise argparse.ArgumentTypeError("token IDs must be non-negative")

    return tokens


def _resolve_dtype(value: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    try:
        return mapping[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported dtype: {value}; choose from {', '.join(mapping)}"
        ) from exc


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"llama-pure-compute {__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Custom CUDA backend: {is_cuda_backend_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).expanduser().resolve()

    if not model_dir.is_dir():
        print(f"error: model directory does not exist: {model_dir}", file=sys.stderr)
        return 2

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "error: CUDA was requested but torch.cuda.is_available() is False",
            file=sys.stderr,
        )
        return 2

    dtype = _resolve_dtype(args.dtype)

    model = LlamaForCausalLM.from_pretrained(
        str(model_dir),
        device=args.device,
        dtype=dtype,
    )

    generator = LlamaGenerator(
        model=model,
        config=model.config,
    )

    print("Generated token IDs:")

    for token_id in generator.generate(
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    ):
        print(token_id, end=" ", flush=True)

    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-pure-compute",
        description="High-performance CUDA/Triton Llama inference runtime.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="report Python, PyTorch, CUDA, GPU, and backend information",
    )
    doctor.set_defaults(func=cmd_doctor)

    generate = subparsers.add_parser(
        "generate",
        help="run autoregressive generation from token IDs",
    )
    generate.add_argument(
        "--model-dir",
        required=True,
        help="directory containing config.json and model weights",
    )
    generate.add_argument(
        "--prompt-tokens",
        required=True,
        type=_parse_prompt_tokens,
        help="comma-separated input token IDs",
    )
    generate.add_argument(
        "--device",
        default="cuda",
        help="execution device, default: cuda",
    )
    generate.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    generate.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    generate.add_argument(
        "--top-k",
        type=int,
        default=50,
    )
    generate.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    generate.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
    )
    generate.set_defaults(func=cmd_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))