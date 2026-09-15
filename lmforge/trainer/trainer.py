import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from lmforge.trainer.checkpoint import load_checkpoint, save_checkpoint
from lmforge.trainer.config import TrainConfig
from lmforge.trainer.scheduler import build_scheduler
from lmforge.utils.distributed import all_reduce_mean, get_world_size, is_main


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]
        use_amp = device.type == "cuda" and self.dtype != torch.float32
        # 权重仍 fp32，前向用 bf16；MPS 上 autocast 不稳定，不走这条
        self.autocast = (
            torch.autocast(device_type=device.type, dtype=self.dtype)
            if use_amp
            else nullcontext()
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = build_scheduler(
            self.optimizer,
            config.warmup_steps,
            config.max_steps,
            config.lr,
            config.min_lr,
        )
        self.step = 0
        if config.resume:
            self.step = load_checkpoint(config.resume, model, self.optimizer, self.scheduler)

    def _build_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay, seen = [], [], set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param.ndim < 2 or "norm" in name or "embed" in name or "lm_head" in name:
                # 1D 向量、Norm、embedding 不做 weight decay，避免把尺度参数往 0 拉
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            fused=self.device.type == "cuda",
        )

    def _ckpt_extra(self) -> dict:
        cfg = self.config
        if not cfg.lora:
            return {"lora": None}
        return {"lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha}}

    def _batches(self):
        while True:
            for batch in self.dataloader:
                yield batch

    def train(self) -> None:
        cfg = self.config
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        t0 = time.time()
        accum = 0

        for batch in self._batches():
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with self.autocast:
                _, loss = self.model(**batch)
                loss = loss / cfg.grad_accum_steps
            sync = accum + 1 >= cfg.grad_accum_steps
            # 未满 accum 时 DDP no_sync，避免每 micro-batch 都 all-reduce
            if sync or not hasattr(self.model, "no_sync"):
                loss.backward()
            else:
                with self.model.no_sync():
                    loss.backward()
            running_loss += loss.item()
            accum += 1
            if not sync:
                continue

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            accum = 0

            if self.step % cfg.log_interval == 0:
                dt = time.time() - t0
                avg_loss = all_reduce_mean(running_loss / cfg.log_interval, self.device)
                tokens = (
                    cfg.micro_batch_size
                    * batch["input_ids"].size(1)
                    * cfg.grad_accum_steps
                    * cfg.log_interval
                    * get_world_size()
                )
                if is_main():
                    print(
                        f"step {self.step}/{cfg.max_steps}  "
                        f"loss {avg_loss:.4f}  "
                        f"lr {self.scheduler.get_last_lr()[0]:.2e}  "
                        f"tok/s {tokens / dt:.0f}",
                        flush=True,
                    )
                running_loss = 0.0
                t0 = time.time()

            if self.step % cfg.save_interval == 0:
                save_checkpoint(
                    f"{cfg.output_dir}/step_{self.step}.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                    extra=self._ckpt_extra(),
                )
            if self.step >= cfg.max_steps:
                break

        save_checkpoint(
            f"{cfg.output_dir}/last.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            extra=self._ckpt_extra(),
        )
