import torch
import torch.nn as nn
from typing import Optional


class Sampler(nn.Module):
    """
    High-Performance GPU Sampler for Llama_Pure_Compute inference engine.
    Supports Temperature scaling, Top-K, Top-P (Nucleus), and Logit Penalties.
    """

    def __init__(
        self,
        default_temperature: float = 1.0,
        default_top_p: float = 1.0,
        default_top_k: int = 0,
    ):
        super().__init__()
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_top_k = default_top_k

    @torch.inference_mode()
    def apply_penalties(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> torch.Tensor:
        """
        Applies Repetition, Presence, and Frequency penalties in a vectorized batch operator.
        
        Args:
            logits: [batch_size, vocab_size]
            input_ids: [batch_size, seq_len]
        """
        if repetition_penalty == 1.0 and presence_penalty == 0.0 and frequency_penalty == 0.0:
            return logits

        # Ensure input_ids is int64 (long) for PyTorch CUDA indexing ops
        input_ids = input_ids.long()
        batch_size, vocab_size = logits.shape
        # Non-in-place updates to protect original logits memory
        out_logits = logits.clone()

        # Repetition Penalty first on original logits
        if repetition_penalty != 1.0:
            score = torch.gather(out_logits, 1, input_ids)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            out_logits = out_logits.scatter(1, input_ids, score)
        
        # Count based Penalties
        # Compute token frequencies per sequence in batch
        if presence_penalty != 0.0 or frequency_penalty != 0.0:
            counts = torch.zeros((batch_size, vocab_size), dtype=torch.int32, device=logits.device)
            counts.scatter_add_(1, input_ids, torch.ones_like(input_ids, dtype=torch.int32))
            
        # Presence Penalty: Subtract penalty if token appeared at least once
        if presence_penalty != 0.0:
            out_logits = out_logits - (counts > 0).to(out_logits.dtype) * presence_penalty

        # Frequency Penalty: Subtract penalty scaled by token occurrences
        if frequency_penalty != 0.0:
            out_logits = out_logits - counts.to(out_logits.dtype) * frequency_penalty

        return out_logits

    @torch.inference_mode()
    def sample(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        input_ids: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> torch.Tensor:
        """
        Executes sampling over logits.

        Returns:
            next_tokens: [batch_size, 1] (int64 CUDA Tensor)
        """
        temperature = temperature if temperature is not None else self.default_temperature
        top_k = top_k if top_k is not None else self.default_top_k
        top_p = top_p if top_p is not None else self.default_top_p

        # 1. Apply Logit Penalties
        if input_ids is not None:
            logits = self.apply_penalties(
                logits,
                input_ids,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

        logits_original = logits.clone()
        
        # 2. Deterministic Greedy Decoding (temperature <= 0 or top_k=1)
        if temperature <= 0.0 or (top_k == 1 and top_p == 1.0):
            return torch.argmax(logits, dim=-1, keepdim=True)

        # 3. Temperature Scaling
        if temperature != 1.0:
            logits = logits / temperature

        # 4. Top-K Masking
        if top_k > 0 and top_k < logits.size(-1):
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # 5. Top-P (Nucleus) Masking
        if top_p < 1.0 and top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            softmax_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(softmax_probs, dim=-1)

            # Shift cumulative probabilities to keep the first token above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter removed mask back to original logit layout
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        num_valid = (logits > float("-inf")).all()
        if(num_valid == 0).any():
            return torch.argmax(logits_original, dim=1, keepdim=True) 
        
        
        # 6. Categorical Sampling from Softmax Distribution
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens


# Quick Verification Harness
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, vocab_size = 4, 32000

    sampler = Sampler().to(device)
    dummy_logits = torch.randn(batch_size, vocab_size, device=device)
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, 64), device=device)

    # Testing Nucleus + Penalty Sampling
    tokens = sampler.sample(
        logits=dummy_logits,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        input_ids=dummy_input_ids,
        repetition_penalty=1.1,
    )

    print(f"Sampled Output Tensor Shape: {tokens.shape}")
    print(f"Sampled Token IDs: {tokens.squeeze(-1).tolist()}")