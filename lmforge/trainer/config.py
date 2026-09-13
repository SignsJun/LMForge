from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    data_path: str = "data/train.jsonl"
    output_dir: str = "outputs/pretrain"
    max_steps: int = 1000
    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    log_interval: int = 10
    save_interval: int = 500
    seed: int = 42
    num_workers: int = 2
    resume: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
