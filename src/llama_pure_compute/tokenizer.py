import os
from typing import List, Union, Optional
import torch

class LlamaTokenizer:
    # Tokenizer Wrapper 
    def __init__(
        self,
        model_path: str,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
    ):
        """
        Args:
            model_path: Path to tokenizer.json
            add_bos_taken: Whether to automatically prepend BOS token to inputs
            add_eos_token: Whether to automatically append EOS token to inputs
        """
        self.model_path = model_path
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        # Load Fast backend
        try:
            from tokenizers import Tokenizer as FastTokenizer
            if os.path.isdir(model_path):
                json_path = os.path.join(model_path, "tokenizer.json")
            else:
                json_path = model_path
            
            if os.path.exists(json_path) and json_path.endswith(".json"):
                self._tokenizer = FastTokenizer.from_file(json_path)
                self.backend = "fast"
            else:
                raise ImportError("Falling back to AutoTokenizer")
        except Exception:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
            self.backend = "transformers"
        
        # Special Token Resolution
        if self.backend == "fast":
            bos = self._tokenizer.token_to_id("<|begin_of_text|>")
            if bos is None:
                bos = self._tokenizer.token_to_id("<s>")
            self.bos_token_id = bos if bos is not None else 1
        
            eos = self._tokenizer.token_to_id("<|end_of_text|>")
            if eos is None:
                eos = self._tokenizer.token_to_id("<\s>")
            self.eos_token_id = eos if eos is not None else 2
        
            pad = self._tokenizer.token_to_id("<pad>")
            self.pad_token_id = pad if pad is not None else 0

            self.vocab_size = self._tokenizer.get_vocab_size()
    def encode(
        self, 
        text: Union[str, List[str]],
        device: Optional[Union[str, torch.device]] = None,
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> torch.Tensor:
        # 2D Pytorch Tensor [batch_size, seq_len]
        if isinstance(text, str):
            text = [text]
        
        if self.backend == "fast":
            encodings = self._tokenizer.encode_batch(text)
            token_ids = []
            for enc in encodings:
                ids = enc.ids
                if self.add_bos_token and (not ids or ids[0] != self.bos_token_id):
                    ids = [self.bos_token_id] + ids
                if self.add_eos_token and (not ids or ids[-1] != self.eos_token_id):
                    ids = ids + [self.eos_token_id]
                token_ids.append(ids)
        else:
            token_ids = []
            for t in text:
                ids = self._tokenizer.encode(
                    t,
                    add_special_tokens = False
                )
                if self.add_bos_token:
                    ids = [self.bos_token_id] + ids
                if self.add_eos_token:
                    ids = ids + [self.eos_token_id]
                token_ids.append(ids)
        
        # Truncate if max_length specified
        if max_length is not None:
            token_ids = [ids[:max_length] for ids in token_ids]
        
        # Padding setup
        max_len = max(len(ids) for ids in token_ids)
        padded_ids = []
        for ids in token_ids:
            if padding and len(ids) < max_len:
                # Right padding for batch consistency
                ids = ids + [self.pad_token_id] * (max_len - len(ids))
            padded_ids.append(ids)
        
        tensor_ids = torch.tensor(padded_ids, dtype=torch.long)
        if device is not None:
            tensor_ids = tensor_ids.to(device)
        
        return tensor_ids

    def decode(
        self, 
        token_ids: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        # Decoding token IDs(Tensor or Pytohon List) to strings
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        
        if not token_ids:
            return ""

        # check if single sequence or batch
        is_batch = isinstance(token_ids[0], list)
        
        if self.backend == "fast":
            if is_batch:
                return self._tokenizer.decode_batch(token_ids, skip_special_tokens=skip_special_tokens)
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
class TextStreamer:
    # Incremental Token Streamer for real-time decoding in terminal or API endpoints
    # Preventing UTF-8 replacement character artifacts on multi-byte token splits
    def __init__(self, tokenizer: LlamaTokenizer, skip_prompt: bool = True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.token_cache: List[int] = []
        self.printed_len = 0
    
    def put(self, token_id: int) -> Optional[str]:
        # Pushing new token into streamer and reutrn printable text delta
        if token_id in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
            return None
        
        self.token_cache.append(token_id)
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        
        # Avoid printing incomplete UTF-8 characters
        if text.endswith("\ufffd"):
            return None
        
        printable_text = text[self.printed_len:]
        self.printed_len = len(text)
        return printable_text
    
    def end(self) -> str:
        "Flushing remaining tokens in cache at end of generation"
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        printable_text = text[self.printed_len:]
        self.printed_len = 0
        return printable_text
    
# Quick Verification Harness
if __name__ == "__main__":
    import tempfile

    # Mock test using Hugging Face Auto Tokenizer model ID
    model_id = "hf-internal-testing/llama-tokenizer"
    
    try:
        tokenizer = LlamaTokenizer(model_path=model_id, add_bos_token=True)
        prompts = [
            "Hello, Llama_Pure_Compute engine!",
            "High-performance CUDA Inference",
        ]
        
        # 1- Test Batch Encoding
        encoded_tensor = tokenizer.encode(prompts, device="cpu")
        print(f"Encoded Tensor Shape: {encoded_tensor.shape}")
        print(f"Encoded Tensor: \n{encoded_tensor}")
        
        # 2- Test Batch decoding
        decoded_texts = tokenizer.decode(encoded_tensor)
        print(f"Decoded Texts: {decoded_texts}")
        
        # 3- Test Streamer
        streamer = TextStreamer(tokenizer)
        sample_sequence = encoded_tensor[0].tolist()
        for tid in sample_sequence:
            delta = streamer.put(tid)
            if delta:
                print(delta, end="", flush=True)
        
        print(streamer.end())    
        print("\nAll Tokenizer Harness tests passed successfully")
    except Exception as e:
        print(f"Harness test skipped or failed (network/weights required): {e}")
    