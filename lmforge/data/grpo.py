import json
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from lmforge.data.tokenizer import Tokenizer


def _prompt_text(obj: dict) -> tuple[str, str]:
    answer = str(obj.get("answer") or obj.get("gold") or "")
    if obj.get("prompt"):
        return str(obj["prompt"]), answer
    conv = obj.get("conversations") or obj.get("messages") or []
    lines = []
    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant" and not answer:
            answer = content
    lines.append("Assistant:")
    return "\n".join(lines), answer


def _iter_grpo_jsonl(path: Path) -> Iterator[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


class GrpoJsonlDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        max_prompt_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.rank = rank
        self.world_size = world_size

    def _keep(self, idx: int) -> bool:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        stride = self.world_size * num_workers
        return idx % stride == self.rank * num_workers + worker_id

    def __iter__(self):
        for i, obj in enumerate(_iter_grpo_jsonl(self.path)):
            if not self._keep(i):
                continue
            text, answer = _prompt_text(obj)
            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            ids = ids[-self.max_prompt_len :]
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "answer": answer,
            }


def grpo_collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(item["input_ids"].numel() for item in batch)
    ids, mask = [], []
    for item in batch:
        n = max_len - item["input_ids"].numel()
        ids.append(F.pad(item["input_ids"], (n, 0), value=pad_id))
        mask.append(F.pad(item["attention_mask"], (n, 0), value=0))
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(mask),
        "answers": [item["answer"] for item in batch],
    }
