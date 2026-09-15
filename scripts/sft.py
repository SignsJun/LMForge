import argparse

from torch.utils.data import DataLoader
import yaml

from lmforge.data import SFTJsonlDataset, Tokenizer, collate_packed
from lmforge.models import LlamaForCausalLM, ModelConfig
from lmforge.trainer import TrainConfig, Trainer
from lmforge.utils import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    set_seed,
    wrap_ddp,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/125m.yaml")
    parser.add_argument("--data", default="configs/data/sft.yaml")
    parser.add_argument("--train", default="configs/train/sft.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    train_cfg = TrainConfig.from_yaml(args.train)
    if args.data_path:
        train_cfg.data_path = args.data_path
    if args.resume:
        train_cfg.resume = args.resume
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.output_dir:
        train_cfg.output_dir = args.output_dir

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    device = init_distributed()
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this run, but PyTorch could not initialize a CUDA device"
        )
    set_seed(train_cfg.seed + get_rank())

    model_cfg = ModelConfig.from_yaml(args.model)
    tokenizer = Tokenizer(data_cfg["tokenizer"])
    if tokenizer.vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} > model vocab {model_cfg.vocab_size}"
        )

    model = wrap_ddp(LlamaForCausalLM(model_cfg).to(device), device)
    dataset = SFTJsonlDataset(
        train_cfg.data_path,
        tokenizer,
        seq_len=data_cfg["seq_len"],
        rank=get_rank(),
        world_size=get_world_size(),
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        collate_fn=collate_packed,
        num_workers=train_cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    try:
        Trainer(model, loader, train_cfg, device).train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
