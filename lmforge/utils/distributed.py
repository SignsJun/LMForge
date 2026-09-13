import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    return get_rank() == 0


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def init_distributed() -> torch.device:
    # torchrun 会注入 RANK / LOCAL_RANK / WORLD_SIZE；单进程走 CUDA/MPS/CPU
    if get_world_size() > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if get_world_size() == 1:
        return model
    return DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / get_world_size()).item()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
