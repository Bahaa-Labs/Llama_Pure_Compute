from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch


TokenInput = Union[str, Sequence[str]]
TokenIds = Union[
    Sequence[int],
    Sequence[Sequence[int]],
    torch.Tensor,
]


@dataclass(frozen=True)
class TokenizerOutput:
    """
    Batched tokenizer result.

    input_ids:
        [batch, sequence]

    attention_mask:
        [batch, sequence]
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class LlamaTokenizer:
    """
    Production tokenizer adapter for Llama_Pure_Compute.

    Backend priority:
        1. tokenizers.Tokenizer from tokenizer.json
        2. Hugging Face AutoTokenizer

    The runtime always receives:
        input_ids       -> [B, S]
        attention_mask  -> [B, S]
    """

    def __init__(
        self,
        model_path: str,
        *,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        padding_side: str = "right",
    ) -> None:
        if padding_side not in ("left", "right"):
            raise ValueError(
                "padding_side must be 'left' or 'right'."
            )

        self.model_path = model_path
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.padding_side = padding_side

        self.backend = ""
        self._tokenizer = None

        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        self.pad_token_id: Optional[int] = None
        self.vocab_size: int = 0

        self._load(model_path)
        self._resolve_special_tokens()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, model_path: str) -> None:
        json_path = (
            os.path.join(model_path, "tokenizer.json")
            if os.path.isdir(model_path)
            else model_path
        )

        # First try the native tokenizers backend.
        if (
            os.path.isfile(json_path)
            and json_path.endswith(".json")
        ):
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(
                    json_path
                )

                self.backend = "tokenizers"
                return

            except Exception:
                # Continue to Hugging Face fallback.
                pass

        # Hugging Face fallback.
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Unable to load tokenizer. Install either "
                "'tokenizers' or 'transformers'."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )

        self.backend = "transformers"

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def _resolve_special_tokens(self) -> None:
        if self.backend == "tokenizers":
            tokenizer = self._tokenizer

            bos_candidates = (
                "<|begin_of_text|>",
                "<s>",
                "<bos>",
            )

            eos_candidates = (
                "<|end_of_text|>",
                "</s>",
                "<eos>",
            )

            pad_candidates = (
                "<pad>",
                "<|pad|>",
            )

            self.bos_token_id = self._find_token_id(
                tokenizer,
                bos_candidates,
            )

            self.eos_token_id = self._find_token_id(
                tokenizer,
                eos_candidates,
            )

            self.pad_token_id = self._find_token_id(
                tokenizer,
                pad_candidates,
            )

            self.vocab_size = tokenizer.get_vocab_size()

        else:
            tokenizer = self._tokenizer

            self.bos_token_id = (
                tokenizer.bos_token_id
            )

            self.eos_token_id = (
                tokenizer.eos_token_id
            )

            self.pad_token_id = (
                tokenizer.pad_token_id
            )

            # Some Llama tokenizers intentionally have no pad token.
            if self.pad_token_id is None:
                self.pad_token_id = self.eos_token_id

            self.vocab_size = len(
                tokenizer.get_vocab()
            )

        if self.bos_token_id is None:
            raise RuntimeError(
                "Tokenizer does not define a BOS token."
            )

        if self.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer does not define an EOS token."
            )

        if self.pad_token_id is None:
            raise RuntimeError(
                "Tokenizer does not define a PAD token."
            )

    @staticmethod
    def _find_token_id(
        tokenizer,
        candidates: Sequence[str],
    ) -> Optional[int]:
        for token in candidates:
            token_id = tokenizer.token_to_id(
                token
            )

            if token_id is not None:
                return token_id

        return None

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: TokenInput,
        *,
        device: Optional[Union[str, torch.device]] = None,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
    ) -> Union[
        torch.Tensor,
        TokenizerOutput,
    ]:
        """
        Encode one or more strings.

        Returns:
            TokenizerOutput when return_attention_mask=True.
            Otherwise input_ids tensor.
        """

        texts = (
            [text]
            if isinstance(text, str)
            else list(text)
        )

        if not texts:
            raise ValueError(
                "text must contain at least one string."
            )

        token_ids: list[list[int]] = []

        for value in texts:
            if not isinstance(value, str):
                raise TypeError(
                    "All text inputs must be strings."
                )

            ids = self._encode_single(
                value
            )

            if self.add_bos_token:
                if (
                    not ids
                    or ids[0]
                    != self.bos_token_id
                ):
                    ids.insert(
                        0,
                        self.bos_token_id,
                    )

            if self.add_eos_token:
                if (
                    not ids
                    or ids[-1]
                    != self.eos_token_id
                ):
                    ids.append(
                        self.eos_token_id,
                    )

            # Truncation happens after special-token insertion.
            if (
                max_length is not None
                and truncation
                and len(ids) > max_length
            ):
                ids = ids[:max_length]

            if not ids:
                raise ValueError(
                    "Tokenizer produced an empty sequence."
                )

            token_ids.append(ids)

        if padding:
            padded, attention = self._pad(
                token_ids
            )
        else:
            if not all(
                len(ids) == len(token_ids[0])
                for ids in token_ids
            ):
                raise ValueError(
                    "padding=False requires equal-length "
                    "sequences."
                )

            padded = token_ids
            attention = [
                [1] * len(ids)
                for ids in token_ids
            ]

        input_ids = torch.tensor(
            padded,
            dtype=torch.long,
        )

        attention_mask = torch.tensor(
            attention,
            dtype=torch.long,
        )

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(
                device
            )

        if return_attention_mask:
            return TokenizerOutput(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return input_ids

    def _encode_single(
        self,
        text: str,
    ) -> list[int]:

        if self.backend == "tokenizers":
            encoding = self._tokenizer.encode(
                text,
            )
            return list(encoding.ids)

        return list(
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------

    def _pad(
        self,
        sequences: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:

        max_len = max(
            len(sequence)
            for sequence in sequences
        )

        padded: list[list[int]] = []
        masks: list[list[int]] = []

        for sequence in sequences:
            pad_count = max_len - len(
                sequence
            )

            if self.padding_side == "right":
                padded.append(
                    sequence
                    + [self.pad_token_id] * pad_count
                )

                masks.append(
                    [1] * len(sequence)
                    + [0] * pad_count
                )

            else:
                padded.append(
                    [self.pad_token_id] * pad_count
                    + sequence
                )

                masks.append(
                    [0] * pad_count
                    + [1] * len(sequence)
                )

        return padded, masks

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        token_ids: TokenIds,
        *,
        skip_special_tokens: bool = True,
    ) -> Union[str, list[str]]:

        if isinstance(
            token_ids,
            torch.Tensor,
        ):
            token_ids = (
                token_ids.detach()
                .cpu()
                .tolist()
            )

        if not token_ids:
            return ""

        is_batch = (
            isinstance(token_ids[0], list)
        )

        if self.backend == "tokenizers":
            if is_batch:
                return self._tokenizer.decode_batch(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                )

            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )

        if is_batch:
            return self._tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bos_token(self) -> int:
        return self.bos_token_id

    @property
    def eos_token(self) -> int:
        return self.eos_token_id

    @property
    def pad_token(self) -> int:
        return self.pad_token_id


class TextStreamer:
    """
    Incremental text streamer.

    The streamer avoids repeatedly decoding the complete history by retaining
    token state internally and only emitting newly decodable text.
    """

    def __init__(
        self,
        tokenizer: LlamaTokenizer,
        *,
        skip_prompt: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt

        self._token_cache: list[int] = []
        self._printed_length = 0
        self._prompt_skipped = not skip_prompt

    def reset(self) -> None:
        self._token_cache.clear()
        self._printed_length = 0
        self._prompt_skipped = not self.skip_prompt

    def put(
        self,
        token_id: int,
    ) -> Optional[str]:

        token_id = int(token_id)

        if token_id in (
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ):
            return None

        self._token_cache.append(
            token_id
        )

        text = self.tokenizer.decode(
            self._token_cache,
            skip_special_tokens=True,
        )

        # Delay emission until a byte sequence becomes decodable.
        if text.endswith("\ufffd"):
            return None

        if not self._prompt_skipped:
            self._prompt_skipped = True
            self._printed_length = len(text)
            return None

        delta = text[
            self._printed_length:
        ]

        self._printed_length = len(text)

        return delta or None

    def end(self) -> str:
        text = self.tokenizer.decode(
            self._token_cache,
            skip_special_tokens=True,
        )

        delta = text[
            self._printed_length:
        ]

        self.reset()

        return delta


if __name__ == "__main__":
    import tempfile

    tokenizer_model = (
        "hf-internal-testing/llama-tokenizer"
    )

    try:
        tokenizer = LlamaTokenizer(
            tokenizer_model,
            add_bos_token=True,
            add_eos_token=False,
        )

        prompts = [
            "Hello, Llama_Pure_Compute engine!",
            "High-performance CUDA inference.",
        ]

        encoded = tokenizer.encode(
            prompts,
            device="cpu",
        )

        print(
            "Input IDs:",
            encoded.input_ids,
        )

        print(
            "Attention mask:",
            encoded.attention_mask,
        )

        decoded = tokenizer.decode(
            encoded.input_ids
        )

        print(
            "Decoded:",
            decoded,
        )

        streamer = TextStreamer(
            tokenizer,
            skip_prompt=False,
        )

        for token_id in encoded.input_ids[
            0
        ].tolist():
            delta = streamer.put(
                token_id
            )

            if delta:
                print(
                    delta,
                    end="",
                    flush=True,
                )

        print(
            streamer.end()
        )

        print(
            "\nTokenizer verification passed."
        )

    except Exception as exc:
        print(
            f"Tokenizer verification unavailable: {exc}"
        )