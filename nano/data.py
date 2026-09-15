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


def encode_sft_sample(
    tokenizer: Tokenizer,
    conversations: list[dict],
    seq_len: int,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    input_ids: list[int] = []
    labels: list[int] = []

    for i, msg in enumerate(conversations):
        # mask 掉user 或者 system 的 content，他们仅用作prompt，不进行loss 计算
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "system":
            ids = tokenizer.encode(f"System: {content}\n")
            input_ids += ids
            labels += [-100] * len(ids)
            continue
        if role == "user":
            ids = tokenizer.encode(f"User: {content}\n")
            input_ids += ids
            labels += [-100] * len(ids)
            continue
        if role != "assistant":
            continue

        header_ids = tokenizer.encode("Assistant:")
        body_ids = tokenizer.encode(f" {content}") if content else []
        turn_ids = header_ids + body_ids + [eos_id]
        turn_labels = [-100] * len(header_ids) + body_ids + [eos_id]
        # 如果这不是最后一个 turn，则添加一个换行符，并 mask 掉
        if i != len(conversations) - 1:
            nl = tokenizer.encode("\n")
            turn_ids += nl
            turn_labels += [-100] * len(nl)
        input_ids += turn_ids
        labels += turn_labels

    # 截断到 seq_len
    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]
    attn = [1] * len(input_ids)
    pad_n = seq_len - len(input_ids)
    # 如果长度不够，则添加右 padding，并 mask 掉
    if pad_n:
        input_ids += [pad_id] * pad_n
        labels += [-100] * pad_n
        attn += [0] * pad_n
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


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


def _iter_sft_jsonl(path: Path) -> Iterator[list[dict]]:
    # 先排列好所有文件，然后按顺序读取
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                # 去除空格和换行符
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                conv = obj.get("conversations") or obj.get("messages")
                if conv:
                    yield conv


class SFTJsonlDataset(IterableDataset):
    def __init__(self, path: str | Path, tokenizer: Tokenizer, seq_len: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        for conv in _iter_sft_jsonl(self.path):
            sample = encode_sft_sample(self.tokenizer, conv, self.seq_len)
            if (sample["labels"] != -100).any():
                yield sample


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
