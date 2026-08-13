from __future__ import annotations

import pytest
import torch

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.generate import GenerationConfig
from llama_pure_compute.kv_manager import KVCacheManager
from llama_pure_compute.model import LlamaForCausalLM
from llama_pure_compute.runtime import (
    GenerationRequest,
    LlamaInferenceEngine,
)


@pytest.fixture
def tiny_config() -> LlamaModelConfig:
    return LlamaModelConfig(
        vocab_size=256,
        dim=128,
        inter_dim=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=64,
        max_batch_size=1,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_kv_cache_has_one_cache_per_layer(
    tiny_config: LlamaModelConfig,
) -> None:

    cache = KVCacheManager(
        num_layers=tiny_config.num_layers,
        max_batch_size=1,
        max_seq_len=64,
        n_kv_heads=tiny_config.num_kv_heads,
        head_dim=tiny_config.head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    assert len(cache.k_cache) == 2
    assert len(cache.v_cache) == 2

    assert (
        cache.k_cache[0].data_ptr()
        != cache.k_cache[1].data_ptr()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_runtime_generation(
    tiny_config: LlamaModelConfig,
) -> None:

    model = (
        LlamaForCausalLM(
            tiny_config
        )
        .cuda()
        .half()
        .eval()
    )

    engine = LlamaInferenceEngine(
        model
    )

    result = engine.generate(
        GenerationRequest(
            prompt_tokens=[1, 2, 3, 4],
            generation=GenerationConfig(
                max_new_tokens=4,
                temperature=0.0,
            ),
        )
    )

    assert 1 <= len(
        result.token_ids
    ) <= 4

    assert result.metrics.prompt_tokens == 4

    assert (
        result.metrics.generated_tokens
        == len(result.token_ids)
    )

    assert result.metrics.total_latency_ms >= 0

    assert result.metrics.ttft_ms >= 0

    assert result.metrics.tokens_per_second > 0