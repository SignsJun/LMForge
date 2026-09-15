import argparse

from torch.utils.data import DataLoader

from .config import NanoConfig
from .data import CausalJsonlDataset, SFTJsonlDataset, Tokenizer, collate_fn
from .model import LlamaForCausalLM
from .trainer import TrainConfig, Trainer, get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/tinystories.jsonl")
    parser.add_argument("--output-dir", default="outputs/nano")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--sft", action="store_true")
    args = parser.parse_args()

    train_args = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        learning_rate=1e-5 if args.sft else 6e-4,
        min_lr=1e-6 if args.sft else 6e-5,
        warmup_steps=10 if args.sft else 20,
    )
    set_seed(train_args.seed)
    device = get_device()

    tokenizer = Tokenizer(train_args.tokenizer_name)
    config = NanoConfig()
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
    model = LlamaForCausalLM(config).to(device)
    dataset = (
        SFTJsonlDataset(train_args.data_path, tokenizer, train_args.seq_len)
        if args.sft
        else CausalJsonlDataset(train_args.data_path, tokenizer, train_args.seq_len)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_args.micro_batch_size,
        collate_fn=collate_fn,
        num_workers=train_args.num_workers,
    )
    Trainer(model, dataloader, train_args, device).train()


if __name__ == "__main__":
    main()
