from __future__ import annotations

import pytest
import torch
from llama_pure_compute.kv_manager import KVCacheManager
from llama_pure_compute.ops import update_kv_cache, is_cuda_backend_available


# Test Fixtures & Configurations
@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def dtype(request):
    return request.param


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_config():
    return {
        "max_batch_size": 4,
        "max_seq_len": 128,
        "n_kv_heads": 8,
        "head_dim": 64,
    }


# 1. Core Unit Test: Operational Accuracy & Parity
class TestKVCacheOperations:
    def test_single_step_update_correctness(self, dtype, device, sample_config):
        """Validates that update() correctly writes new states into cache memory without precision loss."""
        if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BFloat16 not supported on this CUDA device.")

        manager = KVCacheManager(dtype=dtype, device=device, **sample_config)
        bsz, seq_len = 2, 1
        
        # Random inputs
        k_src = torch.randn((bsz, sample_config["n_kv_heads"], seq_len, sample_config["head_dim"]), dtype=dtype, device=device)
        v_src = torch.randn((bsz, sample_config["n_kv_heads"], seq_len, sample_config["head_dim"]), dtype=dtype, device=device)

        start_pos = 0
        keys_out, values_out = manager.update(k_src, v_src, start_pos=start_pos, seq_len=seq_len)

        # Verify shapes
        assert keys_out.shape == (bsz, sample_config["n_kv_heads"], start_pos + seq_len, sample_config["head_dim"])
        assert values_out.shape == (bsz, sample_config["n_kv_heads"], start_pos + seq_len, sample_config["head_dim"])

        # Verify exact numerical match in cache slice
        torch.testing.assert_close(manager.k_cache[:bsz, :, start_pos:start_pos+seq_len, :], k_src)
        torch.testing.assert_close(manager.v_cache[:bsz, :, start_pos:start_pos+seq_len, :], v_src)

    def test_multi_step_sequential_prefill_and_decode(self, dtype, device, sample_config):
        """Simulates LLM Prefill stage (seq_len > 1) followed by step-by-step Decode stage (seq_len = 1)."""
        if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BFloat16 not supported on this CUDA device.")

        manager = KVCacheManager(dtype=dtype, device=device, **sample_config)
        bsz = 2

        # --- Stage 1: Prefill Step (Length = 16) ---
        prefill_len = 16
        k_prefill = torch.randn((bsz, sample_config["n_kv_heads"], prefill_len, sample_config["head_dim"]), dtype=dtype, device=device)
        v_prefill = torch.randn((bsz, sample_config["n_kv_heads"], prefill_len, sample_config["head_dim"]), dtype=dtype, device=device)

        manager.update(k_prefill, v_prefill, start_pos=0, seq_len=prefill_len)

        # --- Stage 2: 5 Auto-regressive Decoding Steps ---
        curr_pos = prefill_len
        for step in range(5):
            k_decode = torch.randn((bsz, sample_config["n_kv_heads"], 1, sample_config["head_dim"]), dtype=dtype, device=device)
            v_decode = torch.randn((bsz, sample_config["n_kv_heads"], 1, sample_config["head_dim"]), dtype=dtype, device=device)

            keys_out, values_out = manager.update(k_decode, v_decode, start_pos=curr_pos, seq_len=1)
            
            # Assert history is preserved up to total accumulated length
            assert keys_out.shape[-2] == curr_pos + 1
            torch.testing.assert_close(manager.k_cache[:bsz, :, curr_pos:curr_pos+1, :], k_decode)
            torch.testing.assert_close(manager.v_cache[:bsz, :, curr_pos:curr_pos+1, :], v_decode)
            
            curr_pos += 1

def test_custom_slot_mapping_scatter(device, sample_config):
    """Tests non-contiguous scatter updates via explicit slot_mapping tensors (PagedAttention style)."""
    manager = KVCacheManager(dtype=torch.float32, device=device, **sample_config)
    
    num_tokens = 4
    k_src = torch.randn((num_tokens, sample_config["n_kv_heads"], sample_config["head_dim"]), dtype=torch.float32, device=device)
    v_src = torch.randn((num_tokens, sample_config["n_kv_heads"], sample_config["head_dim"]), dtype=torch.float32, device=device)

    # Dispersed slot assignments with an unmapped token (-1) sentinel check
    slot_mapping = torch.tensor([12, 45, -1, 3], dtype=torch.int64, device=device)

    update_kv_cache(
        key_src=k_src,
        value_src=v_src,
        key_cache=manager.k_cache,
        value_cache=manager.v_cache,
        slot_mapping=slot_mapping
    )

    max_seq_len = sample_config["max_seq_len"]

    # Convert linear slot indices to 4D coordinates (b = slot // max_seq_len, s = slot % max_seq_len)
    # Slot 12: b = 12 // max_seq_len, s = 12 % max_seq_len
    b12, s12 = 12 // max_seq_len, 12 % max_seq_len
    b45, s45 = 45 // max_seq_len, 45 % max_seq_len
    b3,  s3  = 3  // max_seq_len, 3  % max_seq_len

    # Assert target slots received k_src values across all heads [B, :, S, :] -> [H, D]
    torch.testing.assert_close(manager.k_cache[b12, :, s12, :], k_src[0])
    torch.testing.assert_close(manager.k_cache[b45, :, s45, :], k_src[1])
    torch.testing.assert_close(manager.k_cache[b3,  :, s3,  :], k_src[3])
    
    # Assert unmapped slot 0 remains untouched (zeros)
    torch.testing.assert_close(manager.k_cache[0, :, 0, :], torch.zeros_like(manager.k_cache[0, :, 0, :]))


# 2. CUDA vs CPU Parity Checks
class TestBackendParity:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for backend parity test")
    def test_cuda_vs_pytorch_fallback_parity(self):
        # Setup test inputs
        bsz, n_heads, seq_len, head_dim = 2, 4, 16, 64
        max_batch_size, max_seq_len = 4, 32
    
        key_states = torch.randn(bsz, n_heads, seq_len, head_dim, dtype=torch.float16)
        value_states = torch.randn(bsz, n_heads, seq_len, head_dim, dtype=torch.float16)
    
        # Random slot mapping test
        num_tokens = bsz * seq_len
        slot_mapping = torch.randperm(max_batch_size * max_seq_len)[:num_tokens].to(torch.int64)

        # CPU Manager Run
        mgr_cpu = KVCacheManager(max_batch_size, max_seq_len, n_heads, head_dim, device="cpu", dtype=torch.float16)
        mgr_cpu.update(key_states, value_states, start_pos=0, seq_len=seq_len, slot_mapping=slot_mapping)

        # CUDA Manager Run
        mgr_cuda = KVCacheManager(max_batch_size, max_seq_len, n_heads, head_dim, device="cuda", dtype=torch.float16)
        mgr_cuda.update(key_states.cuda(), value_states.cuda(), start_pos=0, seq_len=seq_len, slot_mapping=slot_mapping.cuda())

        # Verify parity across entire 4D cache allocation
        torch.testing.assert_close(mgr_cpu.k_cache, mgr_cuda.k_cache.cpu())
        torch.testing.assert_close(mgr_cpu.v_cache, mgr_cuda.v_cache.cpu())


# 3. Memory Lifecycle & Boundaries
class TestCacheLifecycleAndBounds:
    def test_reset_clears_memory_buffers(self, device, sample_config):
        """Verifies reset() zeroes cache storage cleanly without requiring memory re-allocation."""
        manager = KVCacheManager(device=device, **sample_config)
        
        # Fill cache with non-zero values
        manager.k_cache.fill_(5.5)
        manager.v_cache.fill_(5.5)

        manager.reset()

        assert torch.all(manager.k_cache == 0.0)
        assert torch.all(manager.v_cache == 0.0)

    def test_batch_size_overflow_raises_error(self, device, sample_config):
        """Ensures appropriate error handling when input exceeds pre-allocated batch size."""
        manager = KVCacheManager(device=device, **sample_config)
        overflow_bsz = sample_config["max_batch_size"] + 1

        k_overflow = torch.randn((overflow_bsz, sample_config["n_kv_heads"], 1, sample_config["head_dim"]), device=device)
        v_overflow = torch.randn((overflow_bsz, sample_config["n_kv_heads"], 1, sample_config["head_dim"]), device=device)

        with pytest.raises(ValueError, match="exceeds maximum pre-allocated batch size"):
            manager.update(k_overflow, v_overflow, start_pos=0, seq_len=1)

    def test_sequence_length_overflow_raises_error(self, device, sample_config):
        """Ensures appropriate error handling when input position exceeds pre-allocated sequence length."""
        manager = KVCacheManager(device=device, **sample_config)
        start_pos = sample_config["max_seq_len"] - 2
        seq_len = 4  # Exceeds max_seq_len by 2 positions

        k_overflow = torch.randn((1, sample_config["n_kv_heads"], seq_len, sample_config["head_dim"]), device=device)
        v_overflow = torch.randn((1, sample_config["n_kv_heads"], seq_len, sample_config["head_dim"]), device=device)

        with pytest.raises(ValueError, match="exceeds max sequence length"):
            manager.update(k_overflow, v_overflow, start_pos=start_pos, seq_len=seq_len)
