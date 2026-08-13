from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from llama_pure_compute.generate import GenerationConfig
from llama_pure_compute.runtime import (
    GenerationRequest,
    LlamaInferenceEngine,
)

from .metrics import (
    ACTIVE_REQUESTS,
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    TTFT,
    TOKENS_GENERATED,
)


class GenerateRequest(BaseModel):
    prompt_tokens: list[int] = Field(
        min_length=1
    )

    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=4096,
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
    )

    top_k: int = Field(
        default=50,
        ge=0,
    )

    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )

    repetition_penalty: float = Field(
        default=1.0,
        gt=0.0,
    )


class GenerateResponse(BaseModel):
    request_id: str
    token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float


_ENGINE: LlamaInferenceEngine | None = None


def configure_engine(
    engine: LlamaInferenceEngine,
) -> None:
    global _ENGINE
    _ENGINE = engine


def get_engine() -> LlamaInferenceEngine:
    if _ENGINE is None:
        raise RuntimeError(
            "Inference engine has not been configured."
        )

    return _ENGINE


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    yield


app = FastAPI(
    title="Llama_Pure_Compute",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.get("/ready")
async def ready() -> dict[str, str]:
    try:
        get_engine()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    return {
        "status": "ready",
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
)
async def generate(
    request: GenerateRequest,
) -> GenerateResponse:

    engine = get_engine()

    request_id = str(
        uuid.uuid4()
    )

    generation = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    inference_request = GenerationRequest(
        prompt_tokens=request.prompt_tokens,
        generation=generation,
    )

    ACTIVE_REQUESTS.inc()

    started = time.perf_counter()

    try:

        result = await asyncio.to_thread(
            engine.generate,
            inference_request,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        REQUESTS_TOTAL.labels(
            status="success"
        ).inc()

        REQUEST_LATENCY.observe(
            elapsed
        )

        TOKENS_GENERATED.inc(
            len(result.token_ids)
        )

        if result.metrics.ttft_ms:
            TTFT.observe(
                result.metrics.ttft_ms
                / 1000.0
            )

        return GenerateResponse(
            request_id=request_id,
            token_ids=list(
                result.token_ids
            ),
            prompt_tokens=(
                result.metrics.prompt_tokens
            ),
            generated_tokens=(
                result.metrics.generated_tokens
            ),
            ttft_ms=(
                result.metrics.ttft_ms
            ),
            total_latency_ms=(
                result.metrics.total_latency_ms
            ),
            tokens_per_second=(
                result.metrics.tokens_per_second
            ),
        )

    except Exception:

        REQUESTS_TOTAL.labels(
            status="error"
        ).inc()

        raise

    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/v1/generate/stream")
async def generate_stream(
    request: GenerateRequest,
) -> StreamingResponse:

    engine = get_engine()

    generation = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )

    inference_request = GenerationRequest(
        prompt_tokens=request.prompt_tokens,
        generation=generation,
    )

    async def event_stream() -> AsyncGenerator[
        str,
        None,
    ]:

        ACTIVE_REQUESTS.inc()

        try:
            result_queue: asyncio.Queue[
                object
            ] = asyncio.Queue()

            sentinel = object()

            def run_generation() -> None:
                try:
                    result = engine.generate(
                        inference_request
                    )

                    for token in result.token_ids:
                        result_queue.put_nowait(
                            {
                                "token_id": token,
                            }
                        )

                    result_queue.put_nowait(
                        {
                            "done": True,
                            "metrics": {
                                "ttft_ms":
                                    result.metrics.ttft_ms,
                                "tokens_per_second":
                                    result.metrics.tokens_per_second,
                            },
                        }
                    )

                except Exception as exc:
                    result_queue.put_nowait(
                        exc
                    )

                finally:
                    result_queue.put_nowait(
                        sentinel
                    )

            asyncio.create_task(
                asyncio.to_thread(
                    run_generation
                )
            )

            while True:

                item = await result_queue.get()

                if item is sentinel:
                    break

                if isinstance(
                    item,
                    Exception,
                ):
                    raise item

                yield (
                    "data: "
                    + json.dumps(item)
                    + "\n\n"
                )

        finally:
            ACTIVE_REQUESTS.dec()

    return StreamingResponse(
        event_stream(),
        media_type=(
            "text/event-stream"
        ),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )