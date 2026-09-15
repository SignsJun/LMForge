import argparse

from torch.utils.data import DataLoader
import yaml

from lmforge.data import SFTJsonlDataset, Tokenizer, collate_packed
from lmforge.lora import inject_lora, load_base_weights, lora_param_stats
from lmforge.models import LlamaForCausalLM, ModelConfig
from lmforge.trainer import TrainConfig, Trainer
from lmforge.utils import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main,
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
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--base-ckpt", default=None)
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
    if args.lora:
        train_cfg.lora = True
    if args.lora_r is not None:
        train_cfg.lora_r = args.lora_r
    if args.lora_alpha is not None:
        train_cfg.lora_alpha = args.lora_alpha
    if args.base_ckpt:
        train_cfg.base_ckpt = args.base_ckpt
    if train_cfg.lora:
        if train_cfg.lr == 1.0e-5:
            train_cfg.lr = 1.0e-4
        if train_cfg.min_lr == 1.0e-6:
            train_cfg.min_lr = 1.0e-5

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

    model = LlamaForCausalLM(model_cfg)
    if train_cfg.base_ckpt:
        load_base_weights(model, train_cfg.base_ckpt)
    if train_cfg.lora:
        inject_lora(model, r=train_cfg.lora_r, alpha=train_cfg.lora_alpha)
        trainable, total = lora_param_stats(model)
        if is_main():
            print(f"lora trainable {trainable}/{total} ({100 * trainable / total:.2f}%)", flush=True)
    model = wrap_ddp(model.to(device), device)
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
