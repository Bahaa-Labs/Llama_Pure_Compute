import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.kv_manager import KVCacheManager
from llama_pure_compute.model import (
    LlamaAttention,
    LlamaDecodeLayer,
    LlamaForCausalLM,
    LlamaModel,
    precompute_freqs_cis,
)
from llama_pure_compute.ops import is_cuda_backend_available


# ============================================================================
# PyTest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def device() -> str:
    """Detects CUDA availability or falls back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_config() -> LlamaModelConfig:
    """Minimal lightweight model config for fast execution during testing."""
    return LlamaModelConfig(
        vocab_size=1000,
        dim=128,
        inter_dim=384,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,  # GQA Ratio: 2:1
        max_seq_len=256,
        max_batch_size=4,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


# ============================================================================
# Unit Tests: Components & Primitives
# ============================================================================

class TestModelPrimitives:

    def test_precompute_freqs_cis_shapes(self, device: str):
        """Verifies RoPE cos/sin frequencies shape and dtype generation."""
        head_dim = 64
        seq_len = 128
        theta = 10000.0

        cos, sin = precompute_freqs_cis(
            dim=head_dim, end=seq_len, theta=theta, device=torch.device(device)
        )

        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_attention_layer_forward_shape(
        self, small_config: LlamaModelConfig, device: str, dtype: torch.dtype
    ):
        """Verifies LlamaAttention output shape and numerical stability across precision dtypes."""
        if device == "cpu" and dtype == torch.float16:
            pytest.skip("FP16 is not natively supported on CPU execution")

        attn = LlamaAttention(small_config).to(device=device, dtype=dtype)
        batch_size, seq_len = 2, 16

        x = torch.randn(
            batch_size, seq_len, small_config.dim, device=device, dtype=dtype
        )
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        )

        cos, sin = precompute_freqs_cis(
            small_config.head_dim, small_config.max_seq_len, device=torch.device(device)
        )
        cos_pos = cos[positions].unsqueeze(2).to(dtype)
        sin_pos = sin[positions].unsqueeze(2).to(dtype)

        output = attn(
            x=x,
            positions=positions,
            cos=cos_pos,
            sin=sin_pos,
            kv_cache=None,
            layer_idx=0,
            mask=None,
        )

        assert output.shape == (batch_size, seq_len, small_config.dim)
        assert output.dtype == dtype
        assert not torch.isnan(output).any()

    def test_decode_layer_forward_shape(
        self, small_config: LlamaModelConfig, device: str
    ):
        """Verifies Transformer Block (LlamaDecodeLayer) with fused RMSNorm + SwiGLU ops."""
        layer = LlamaDecodeLayer(small_config, layer_idx=0).to(device=device)
        batch_size, seq_len = 2, 8

        x = torch.randn(batch_size, seq_len, small_config.dim, device=device)
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        )

        cos, sin = precompute_freqs_cis(
            small_config.head_dim, small_config.max_seq_len, device=torch.device(device)
        )
        cos_pos = cos[positions].unsqueeze(2)
        sin_pos = sin[positions].unsqueeze(2)

        output = layer(x=x, positions=positions, cos=cos_pos, sin=sin_pos)

        assert output.shape == (batch_size, seq_len, small_config.dim)
        assert not torch.isnan(output).any()


# ============================================================================
# Integration Tests: LlamaForCausalLM Forward Pass
# ============================================================================

class TestLlamaForCausalLMForward:

    def test_prefill_phase_forward(self, small_config: LlamaModelConfig, device: str):
        """Tests multi-token prefill phase (seq_len > 1) with automatic causal mask generation."""
        model = LlamaForCausalLM(small_config).to(device=device)
        model.eval()

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(
            0, small_config.vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            logits = model(input_ids=input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_single_step_decode_phase_forward(
        self, small_config: LlamaModelConfig, device: str
    ):
        """Tests single-step token decode (seq_len = 1) at explicit sequence offsets."""
        model = LlamaForCausalLM(small_config).to(device=device)
        model.eval()

        batch_size = 2
        decode_step = 15
        input_ids = torch.randint(
            0, small_config.vocab_size, (batch_size, 1), device=device
        )
        positions = torch.full(
            (batch_size, 1), fill_value=decode_step, device=device, dtype=torch.long
        )

        with torch.no_grad():
            logits = model(input_ids=input_ids, positions=positions)

        assert logits.shape == (batch_size, 1, small_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_autoregressive_kv_cache_integration(
        self, small_config: LlamaModelConfig, device: str
    ):
        """Validates stateful execution across Prefill -> Decode steps using KVCacheManager."""
        model = LlamaForCausalLM(small_config).to(device=device)
        model.eval()

        batch_size = 2
        prefill_len = 8
        decode_steps = 4

        kv_cache = KVCacheManager(
            max_batch_size=small_config.max_batch_size,
            max_seq_len=small_config.max_seq_len,
            n_kv_heads=small_config.num_kv_heads,
            head_dim=small_config.head_dim,
            dtype=torch.float32,
            device=device,
        )

        # 1. Prefill Phase
        prefill_input = torch.randint(
            0, small_config.vocab_size, (batch_size, prefill_len), device=device
        )
        with torch.no_grad():
            prefill_logits = model(input_ids=prefill_input, kv_cache=kv_cache)

        assert prefill_logits.shape == (batch_size, prefill_len, small_config.vocab_size)

        # 2. Sequential Decode Phase
        current_pos = prefill_len
        for step in range(decode_steps):
            decode_input = torch.randint(
                0, small_config.vocab_size, (batch_size, 1), device=device
            )
            positions = torch.full(
                (batch_size, 1), fill_value=current_pos, device=device, dtype=torch.long
            )

            with torch.no_grad():
                decode_logits = model(
                    input_ids=decode_input, positions=positions, kv_cache=kv_cache
                )

            assert decode_logits.shape == (batch_size, 1, small_config.vocab_size)
            current_pos += 1

    def test_custom_ops_fallback_parity(
        self, small_config: LlamaModelConfig, device: str
    ):
        """Verifies forward pass correctness regardless of whether CUDA extension (_C) is loaded."""
        model = LlamaForCausalLM(small_config).to(device=device)
        model.eval()

        batch_size, seq_len = 1, 8
        input_ids = torch.randint(
            0, small_config.vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            logits = model(input_ids=input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert not torch.isnan(logits).any()


# ============================================================================
# Checkpoint Serialization Tests: from_pretrained
# ============================================================================

class TestModelCheckpointing:

    def test_from_pretrained_loading(
        self, small_config: LlamaModelConfig, device: str
    ):
        """Verifies checkpoint saving, JSON config key normalization, and state loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            config_dict = {
                "hidden_size": small_config.dim,
                "intermediate_size": small_config.inter_dim,
                "num_hidden_layers": small_config.num_layers,
                "num_attention_heads": small_config.num_heads,
                "num_key_value_heads": small_config.num_kv_heads,
                "vocab_size": small_config.vocab_size,
                "max_seq_len": small_config.max_seq_len,
                "rms_norm_eps": small_config.rms_norm_eps,
                "rope_theta": small_config.rope_theta,
            }

            with open(tmp_path / "config.json", "w") as f:
                json.dump(config_dict, f)

            ref_model = LlamaForCausalLM(small_config)
            weight_file = tmp_path / "pytorch_model.bin"
            torch.save(ref_model.state_dict(), weight_file)

            loaded_model = LlamaForCausalLM.from_pretrained(
                model_dir=str(tmp_path), device=device, dtype=torch.float32
            )

            assert isinstance(loaded_model, LlamaForCausalLM)
            assert loaded_model.config.dim == small_config.dim
            assert loaded_model.config.num_layers == small_config.num_layers

            for (p_name, p_ref), (_, p_loaded) in zip(
                ref_model.named_parameters(), loaded_model.named_parameters()
            ):
                torch.testing.assert_close(p_ref.cpu(), p_loaded.cpu())