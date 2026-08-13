from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cuda: tests requiring a CUDA-capable NVIDIA GPU",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmark tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if os.environ.get("LLAMA_CI_GPU", "0") == "1":
        return

    skip_cuda = pytest.mark.skip(
        reason="CUDA tests disabled on CPU CI runner",
    )

    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)