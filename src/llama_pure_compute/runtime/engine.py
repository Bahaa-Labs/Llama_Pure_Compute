from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from llama_pure_compute.tokenizer import LlamaTokenizer

import torch

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.generate import (
    GenerationConfig,
    LlamaGenerator,
)
from llama_pure_compute.model import LlamaForCausalLM
from llama_pure_compute.runtime.metrics import GenerationMetrics


@dataclass(frozen=True)
class GenerationRequest:
    prompt_tokens: Sequence[int]
    generation: GenerationConfig = GenerationConfig()


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    metrics: GenerationMetrics


class LlamaInferenceEngine:
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: Optional[LlamaTokenizer] = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer

        self.generator = LlamaGenerator(
            model=self.model,
            config=self.model.config,
        )
    @property
    def config(self) -> LlamaModelConfig:
        return self.model.config

    @torch.inference_mode()
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:

        if not request.prompt_tokens:
            raise ValueError(
                "prompt_tokens must not be empty"
            )

        start = time.perf_counter()

        first_token_time: Optional[float] = None
        output: list[int] = []

        for token_id in self.generator.generate(
            request.prompt_tokens,
            generation=request.generation,
        ):
            now = time.perf_counter()

            if first_token_time is None:
                first_token_time = now

            output.append(int(token_id))

        end = time.perf_counter()

        if first_token_time is None:
            first_token_time = end

        total_latency_ms = (
            end - start
        ) * 1000.0

        ttft_ms = (
            first_token_time - start
        ) * 1000.0

        decode_latency_ms = max(
            total_latency_ms - ttft_ms,
            0.0,
        )

        generated_tokens = len(output)

        tokens_per_second = (
            generated_tokens
            / max(
                end - start,
                1e-12,
            )
        )

        metrics = GenerationMetrics(
            prompt_tokens=len(
                request.prompt_tokens
            ),
            generated_tokens=generated_tokens,
            total_latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            decode_latency_ms=decode_latency_ms,
            tokens_per_second=tokens_per_second,
        )

        return GenerationResult(
            token_ids=tuple(output),
            metrics=metrics,
        )

    def reset(self) -> None:
        self.generator.reset()

@torch.inference_mode()
def generate_text(
    self,
    prompt: str,
    *,
    generation: GenerationConfig = GenerationConfig(),
) -> tuple[str, GenerationMetrics]:

    if self.tokenizer is None:
        raise RuntimeError(
            "generate_text() requires a tokenizer."
        )

    encoded = self.tokenizer.encode(
        prompt,
        device=next(
            self.model.parameters()
        ).device,
        padding=False,
    )

    generated, metrics = self.generate(
        GenerationRequest(
            prompt_tokens=encoded.input_ids[0].tolist(),
            generation=generation,
        )
    )

    text = self.tokenizer.decode(
        generated.token_ids,
        skip_special_tokens=True,
    )

    return text, metrics