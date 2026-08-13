from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Sequence

import torch
import torch.nn.functional as F

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.kv_manager import KVCacheManager


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    eos_token_ids: tuple[int, ...] = ()


def apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    penalty: float,
) -> torch.Tensor:

    if penalty <= 0:
        raise ValueError(
            "repetition_penalty must be > 0"
        )

    if penalty == 1.0 or history.numel() == 0:
        return logits

    scores = torch.gather(
        logits,
        -1,
        history,
    )

    scores = torch.where(
        scores < 0,
        scores * penalty,
        scores / penalty,
    )

    logits.scatter_(
        -1,
        history,
        scores,
    )

    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:

    if temperature < 0:
        raise ValueError(
            "temperature must be >= 0"
        )

    if temperature == 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    logits = logits / temperature

    if top_k > 0:
        k = min(
            top_k,
            logits.shape[-1],
        )

        values = torch.topk(
            logits,
            k,
            dim=-1,
        ).values

        threshold = values[..., -1, None]

        logits = logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

    if not 0 < top_p <= 1:
        raise ValueError(
            "top_p must be in (0, 1]"
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        probs = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative = torch.cumsum(
            probs,
            dim=-1,
        )

        remove = cumulative > top_p

        remove[..., 1:] = remove[
            ...,
            :-1,
        ].clone()

        remove[..., 0] = False

        original_remove = torch.zeros_like(
            remove
        )

        original_remove.scatter_(
            -1,
            sorted_indices,
            remove,
        )

        logits = logits.masked_fill(
            original_remove,
            float("-inf"),
        )

    probs = F.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probs,
        num_samples=1,
    )


class LlamaGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        config: LlamaModelConfig,
    ) -> None:

        self.model = model
        self.config = config

        self.device = next(
            model.parameters()
        ).device

        dtype = next(
            model.parameters()
        ).dtype

        self.kv_cache = KVCacheManager(
            num_layers=config.num_layers,
            max_batch_size=1,
            max_seq_len=config.max_seq_len,
            n_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=self.device,
        )

    def reset(self) -> None:
        self.kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        generation: GenerationConfig,
    ) -> Generator[int, None, None]:

        if not prompt_tokens:
            raise ValueError(
                "prompt_tokens must not be empty"
            )

        prompt_len = len(prompt_tokens)

        if (
            prompt_len
            + generation.max_new_tokens
            > self.config.max_seq_len
        ):
            raise ValueError(
                "Requested generation exceeds "
                "configured max_seq_len."
            )

        self.model.eval()

        self.reset()

        self.kv_cache.begin_request(
            batch_size=1
        )

        prompt = torch.tensor(
            [list(prompt_tokens)],
            dtype=torch.long,
            device=self.device,
        )

        # Pre-allocated token history.
        total_capacity = (
            prompt_len
            + generation.max_new_tokens
        )

        history = torch.empty(
            (
                1,
                total_capacity,
            ),
            dtype=torch.long,
            device=self.device,
        )

        history[:, :prompt_len] = prompt

        # ==============================================================
        # PREFILL
        # ==============================================================

        positions = torch.arange(
            prompt_len,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)

        logits = self.model(
            prompt,
            positions=positions,
            kv_cache=self.kv_cache,
            mask=None,
        )

        next_logits = logits[:, -1, :]

        next_logits = apply_repetition_penalty(
            next_logits,
            history[:, :prompt_len],
            generation.repetition_penalty,
        )

        next_token = sample_next_token(
            next_logits,
            temperature=generation.temperature,
            top_k=generation.top_k,
            top_p=generation.top_p,
        )

        history[:, prompt_len] = next_token

        history_len = prompt_len + 1

        token_id = int(
            next_token.item()
        )

        yield token_id

        if token_id in generation.eos_token_ids:
            return

        # ==============================================================
        # DECODE
        # ==============================================================

        for position in range(
            prompt_len,
            total_capacity - 1,
        ):

            position_ids = torch.tensor(
                [[position]],
                dtype=torch.long,
                device=self.device,
            )

            logits = self.model(
                next_token,
                positions=position_ids,
                kv_cache=self.kv_cache,
                mask=None,
            )

            next_logits = logits[:, -1, :]

            next_logits = apply_repetition_penalty(
                next_logits,
                history[:, :history_len],
                generation.repetition_penalty,
            )

            next_token = sample_next_token(
                next_logits,
                temperature=generation.temperature,
                top_k=generation.top_k,
                top_p=generation.top_p,
            )

            history[:, history_len] = next_token
            history_len += 1

            token_id = int(
                next_token.item()
            )

            yield token_id

            if token_id in generation.eos_token_ids:
                return