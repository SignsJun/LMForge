from pathlib import Path

import torch

from lmforge.utils.distributed import barrier, is_main, unwrap


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> None:
    # 只 rank0 写盘，然后 barrier，防止其它 rank 先去读未写完的文件
    if is_main():
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # DDP 包了一层 module，存裸模型才能单卡 load
                "model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            path,
        )
    barrier()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))
