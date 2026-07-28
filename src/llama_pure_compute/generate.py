"""
Autoregressive inference pipeline with support for Temperature, Top-K, 
Top-P (Nucleus) sampling, Repetition Penalty, and KV-Cache management.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Generator, Tuple
from llama_pure_compute.config import LlamaModelConfig


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """Applies repetition penalty in-place on the output logits."""
    if penalty == 1.0 or generated_tokens.numel() == 0:
        return logits

    # Extract unique tokens present in current generation history per batch
    score = torch.gather(logits, 1, generated_tokens)
    # Apply penalty: if logit < 0, multiply; if logit > 0, divide
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, generated_tokens, score)
    return logits


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9
) -> torch.Tensor:
    """
    Applies temperature, top-k filtering, and top-p (nucleus) masking to logits,
    returning sampled token indices.
    """
    if temperature > 0.0:
        logits = logits / temperature
    else:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 1. Top-K Masking
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
        logits[indices_to_remove] = float("-inf")

    # 2. Top-P (Nucleus) Masking
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Shift the masks to the right to keep the first token above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter sorted mask back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    # Sample from the resulting multinomial distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


class LlamaGenerator:
    """
    High-level generation API managing prompt encoding, KV-Cache prefill,
    and single-token decode loop execution.
    """

    def __init__(self, model: torch.nn.Module, config: LlamaModelConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Generator[int, None, None]:
        """
        Generates tokens autoregressively, yielding token IDs one by one.
        """
        self.model.eval()
        stop_tokens = set(stop_token_ids or [])

        # Prepare initial input tensor [1, seq_len]
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        generated_history = tokens.clone()

        # 1. PREFILL PHASE (Process Prompt Tokens & Warm KV Cache)

        # Forward pass over full prompt tokens
        logits = self.model(tokens, start_pos=0)
        
        # Extract last token logits for sampling
        next_logit = logits[:, -1, :].clone()
        
        if repetition_penalty != 1.0:
            next_logit = apply_repetition_penalty(
                next_logit, generated_history, repetition_penalty
            )

        next_token = sample_top_k_top_p(
            next_logit, temperature=temperature, top_k=top_k, top_p=top_p
        )
        
        token_id = next_token.item()
        yield token_id

        if token_id in stop_tokens:
            return

        generated_history = torch.cat([generated_history, next_token], dim=-1)
        curr_pos = tokens.shape[1]

        # 2. DECODE PHASE (Token-by-Token Autoregressive Loop)
        
        for _ in range(max_new_tokens - 1):
            # Pass only single new token into model using active KV cache slot
            logits = self.model(next_token, start_pos=curr_pos)
            next_logit = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                next_logit = apply_repetition_penalty(
                    next_logit, generated_history, repetition_penalty
                )

            next_token = sample_top_k_top_p(
                next_logit, temperature=temperature, top_k=top_k, top_p=top_p
            )
            
            token_id = next_token.item()
            yield token_id

            if token_id in stop_tokens:
                break

            generated_history = torch.cat([generated_history, next_token], dim=-1)
            curr_pos += 1