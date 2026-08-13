from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


def get_nvcc_flags() -> list[str]:
    flags = [
        "-O3",
        "--std=c++17",
        "--use_fast_math",
        "-lineinfo",
    ]

    if os.environ.get("LLAMA_PURE_COMPUTE_DEBUG", "0") == "1":
        flags.extend(["-G", "-g"])

    return flags


setup(
    ext_modules=[
        CUDAExtension(
            name="llama_pure_compute._C",
            sources=[
                "src/csrc/ops/bindings.cpp",
                "src/csrc/ops/rmsswiglu.cu",
                "src/csrc/ops/rope.cu",
                "src/csrc/ops/kv_cache.cu",
            ],
            include_dirs=[
                str(ROOT / "src" / "csrc" / "include"),
                str(ROOT / "src" / "csrc" / "ops"),
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": get_nvcc_flags(),
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)