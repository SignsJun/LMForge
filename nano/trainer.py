import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .model import LlamaForCausalLM


@dataclass
class TrainConfig:
    data_path: str = "data/tinystories.jsonl"
    tokenizer_name: str = "gpt2"
    output_dir: str = "outputs/nano"
    seq_len: int = 256
    max_steps: int = 200
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    save_steps: int = 100
    seed: int = 42
    num_workers: int = 0
    resume_from_checkpoint: str | None = None
    base_ckpt: str | None = None
    lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: AdamW, cfg: TrainConfig) -> LambdaLR:
    # 学习率衰减策略
    def lr_lambda(step: int) -> float:
        # 预热阶段
        if step < cfg.warmup_steps:
            return max(step, 1) / max(cfg.warmup_steps, 1)
        # 衰减阶段
        progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)) / cfg.learning_rate

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: LlamaForCausalLM,
        dataloader: DataLoader,
        args: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        self.device = device
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = build_scheduler(self.optimizer, args)
        self.global_step = 0
        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint)

    def _build_optimizer(self) -> AdamW:
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "embed" in name or "lm_head" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return AdamW(
            [
                {"params": decay, "weight_decay": self.args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.eps,
            fused=self.device.type == "cuda",
        )

    def _save_checkpoint(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "global_step": self.global_step,
                "lora": (
                    {"r": self.args.lora_r, "alpha": self.args.lora_alpha}
                    if self.args.lora
                    else None
                ),
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.global_step = int(ckpt.get("global_step", 0))

    def train(self) -> None:
        args = self.args
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        t0 = time.time()
        accum = 0
        data = iter(self.dataloader)

        while self.global_step < args.max_steps:
            try:
                batch = next(data)
            except StopIteration:
                data = iter(self.dataloader)
                batch = next(data)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            _, loss, _ = self.model(**batch)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()
            accum += 1
            if accum < args.gradient_accumulation_steps:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            accum = 0

            if self.global_step % args.logging_steps == 0:
                dt = time.time() - t0
                tokens = (
                    args.micro_batch_size
                    * batch["input_ids"].size(1)
                    * args.gradient_accumulation_steps
                    * args.logging_steps
                )
                print(
                    f"step {self.global_step}/{args.max_steps}  "
                    f"loss {running_loss / args.logging_steps:.4f}  "
                    f"lr {self.lr_scheduler.get_last_lr()[0]:.2e}  "
                    f"tok/s {tokens / dt:.0f}",
                    flush=True,
                )
                running_loss = 0.0
                t0 = time.time()

            if self.global_step % args.save_steps == 0:
                self._save_checkpoint(f"{args.output_dir}/checkpoint-{self.global_step}.pt")

        self._save_checkpoint(f"{args.output_dir}/pytorch_model.pt")
