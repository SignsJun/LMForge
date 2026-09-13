import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from lmforge.data.tokenizer import Tokenizer


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


class PackedCausalDataset(IterableDataset):
    """把多条文档拼满 seq_len。每条文档重置 position_ids，并记下 document_ids 供跨文档 mask。"""
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        text_key: str = "text",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_key = text_key
        self.rank = rank
        self.world_size = world_size

    def _keep(self, idx: int) -> bool:
        # DDP × DataLoader worker 两层切分，避免各进程读到同一条文档
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        stride = self.world_size * num_workers
        return idx % stride == self.rank * num_workers + worker_id

    def __iter__(self):
        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id
        seq_len = self.seq_len
        ids_buf: list[int] = []
        pos_buf: list[int] = []
        doc_buf: list[int] = []
        doc_idx = 0

        def flush():
            nonlocal ids_buf, pos_buf, doc_buf, doc_idx
            if not ids_buf:
                return None
            # 文档首 token 没有上文可预测，pad 不计入 loss
            labels = [-100 if pos == 0 else tok for tok, pos in zip(ids_buf, pos_buf)]
            pad_n = seq_len - len(ids_buf)
            if pad_n:
                ids_buf += [pad_id] * pad_n
                pos_buf += [0] * pad_n
                doc_buf += [-1] * pad_n
                labels += [-100] * pad_n
            sample = {
                "input_ids": torch.tensor(ids_buf, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "position_ids": torch.tensor(pos_buf, dtype=torch.long),
                "document_ids": torch.tensor(doc_buf, dtype=torch.long),
            }
            ids_buf, pos_buf, doc_buf = [], [], []
            doc_idx = 0
            return sample

        for i, text in enumerate(_iter_jsonl(self.path, self.text_key)):
            if not self._keep(i):
                continue
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue
            tokens = tokens + [eos_id]
            offset = 0
            while offset < len(tokens):
                remaining = seq_len - len(ids_buf)
                chunk = tokens[offset : offset + remaining]
                offset += len(chunk)
                for pos, tok in enumerate(chunk):
                    ids_buf.append(tok)
                    pos_buf.append(pos)
                    doc_buf.append(doc_idx)
                doc_idx += 1
                if len(ids_buf) == seq_len:
                    yield flush()

        sample = flush()
        if sample is not None:
            yield sample


def collate_packed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
