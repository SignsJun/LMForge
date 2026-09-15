import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from lmforge.data.tokenizer import Tokenizer


def encode_sft_sample(
    tokenizer: Tokenizer,
    conversations: list[dict],
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """只对 assistant 回复算 loss；system / user / 角色前缀 / pad 为 -100。"""
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    input_ids: list[int] = []
    labels: list[int] = []

    for i, msg in enumerate(conversations):
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
        if i != len(conversations) - 1:
            nl = tokenizer.encode("\n")
            turn_ids += nl
            turn_labels += [-100] * len(nl)
        input_ids += turn_ids
        labels += turn_labels

    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]
    attn = [1] * len(input_ids)
    pad_n = seq_len - len(input_ids)
    if pad_n:
        input_ids += [pad_id] * pad_n
        labels += [-100] * pad_n
        attn += [0] * pad_n
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def _iter_sft_jsonl(path: Path) -> Iterator[list[dict]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                conv = obj.get("conversations") or obj.get("messages")
                if conv:
                    yield conv


class SFTJsonlDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def _keep(self, idx: int) -> bool:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        stride = self.world_size * num_workers
        return idx % stride == self.rank * num_workers + worker_id

    def __iter__(self):
        for i, conv in enumerate(_iter_sft_jsonl(self.path)):
            if not self._keep(i):
                continue
            sample = encode_sft_sample(self.tokenizer, conv, self.seq_len)
            if (sample["labels"] != -100).any():
                yield sample
