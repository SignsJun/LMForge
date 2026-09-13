import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, name_or_path: str = "gpt2"):
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def pad_token_id(self) -> int:
        return self.tok.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


def _iter_jsonl(path: Path, text_key: str) -> Iterator[str]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj[text_key] if isinstance(obj, dict) else str(obj)
                if text:
                    yield text


class CausalJsonlDataset(IterableDataset):
    def __init__(self, path: str | Path, tokenizer: Tokenizer, seq_len: int, text_key: str = "text"):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key

    def __iter__(self):
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        seq_len = self.seq_len
        buf: list[int] = []

        def flush(chunk: list[int]):
            n = len(chunk)
            labels = chunk.copy()
            attn = [1] * n
            if n < seq_len:
                chunk = chunk + [pad_id] * (seq_len - n)
                labels = labels + [-100] * (seq_len - n)
                attn = attn + [0] * (seq_len - n)
            return {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }

        for text in _iter_jsonl(self.path, self.text_key):
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue
            buf.extend(tokens + [eos_id])
            while len(buf) >= seq_len:
                yield flush(buf[:seq_len])
                buf = buf[seq_len:]
        if buf:
            yield flush(buf)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
