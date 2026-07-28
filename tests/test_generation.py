"""
tests/test_generation.py

Unit tests for LlamaGenerator prefill, decoding shapes, and sampling logic.
"""

import torch
import pytest
from torch import nn
from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.generate import LlamaGenerator, sample_top_k_top_p, apply_repetition_penalty


class DummyLlamaForTesting(nn.Module):
    """Mock Llama model replicating expected tensor shapes for testing generation."""
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Handle configuration field variation (dim vs hidden_dim vs hidden_size)
        hidden_dim = getattr(config, "hidden_dim", getattr(config, "dim", getattr(config, "hidden_size", 256)))
        
        self.embed = nn.Embedding(config.vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        hidden = self.embed(tokens)
        logits = self.head(hidden)
        return logits


@pytest.fixture
def model_and_config():
    # Instantiate with attributes supported by LlamaModelConfig (falling back safely if args vary)
    try:
        config = LlamaModelConfig(
            vocab_size=1024,
            dim=256,
            num_layers=2,
            num_heads=8,
            head_dim=32,
            max_seq_len=512
        )
    except TypeError:
        try:
            config = LlamaModelConfig(
                vocab_size=1024,
                hidden_size=256,
                num_layers=2,
                num_heads=8,
                head_dim=32,
                max_seq_len=512
            )
        except TypeError:
            config = LlamaModelConfig()
            config.vocab_size = 1024
            config.hidden_dim = 256
            config.num_layers = 2
            config.num_heads = 8
            config.head_dim = 32
            config.max_seq_len = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyLlamaForTesting(config).to(device)
    return model, config


def test_sampling_top_k_top_p():
    vocab_size = 100
    logits = torch.randn(1, vocab_size, dtype=torch.float32)
    
    token = sample_top_k_top_p(logits, temperature=0.8, top_k=10, top_p=0.9)
    assert token.shape == (1, 1)
    assert 0 <= token.item() < vocab_size


def test_repetition_penalty():
    vocab_size = 50
    logits = torch.zeros(1, vocab_size, dtype=torch.float32)
    logits[0, 5] = 5.0
    
    generated_tokens = torch.tensor([[5, 5, 5]], dtype=torch.long)
    penalized_logits = apply_repetition_penalty(logits.clone(), generated_tokens, penalty=1.5)
    
    assert penalized_logits[0, 5] < 5.0


def test_llama_generator_integration(model_and_config):
    model, config = model_and_config
    generator = LlamaGenerator(model, config)
    
    prompt = [10, 20, 30, 40, 50]
    max_new_tokens = 10
    
    token_stream = list(
        generator.generate(
            prompt_tokens=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=20,
            top_p=0.95
        )
    )
    
    assert len(token_stream) == max_new_tokens
    for tid in token_stream:
        assert isinstance(tid, int)
        assert 0 <= tid < config.vocab_size