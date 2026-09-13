import argparse

from torch.utils.data import DataLoader

from .config import NanoConfig
from .data import CausalJsonlDataset, Tokenizer, collate_fn
from .model import LlamaForCausalLM
from .trainer import TrainConfig, Trainer, get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/tinystories.jsonl")
    parser.add_argument("--output-dir", default="outputs/nano")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    train_args = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    set_seed(train_args.seed)
    device = get_device()

    tokenizer = Tokenizer(train_args.tokenizer_name)
    config = NanoConfig()
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
    model = LlamaForCausalLM(config).to(device)
    dataset = CausalJsonlDataset(train_args.data_path, tokenizer, train_args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=train_args.micro_batch_size,
        collate_fn=collate_fn,
        num_workers=train_args.num_workers,
    )
    Trainer(model, dataloader, train_args, device).train()


if __name__ == "__main__":
    main()
