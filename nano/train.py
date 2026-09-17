import argparse

from torch.utils.data import DataLoader

from .config import NanoConfig
from .data import CausalJsonlDataset, SFTJsonlDataset, Tokenizer, collate_fn
from .grpo import GrpoJsonlDataset, GrpoTrainer, build_ref_model, grpo_collate
from .lora import inject_lora, load_base_weights, lora_param_stats
from .model import LlamaForCausalLM
from .trainer import TrainConfig, Trainer, get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/tinystories.jsonl")
    parser.add_argument("--output-dir", default="outputs/nano")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    args = parser.parse_args()

    if args.lora or args.grpo:
        lr, min_lr, warmup = 1e-4, 1e-5, 10
    elif args.sft:
        lr, min_lr, warmup = 1e-5, 1e-6, 10
    else:
        lr, min_lr, warmup = 6e-4, 6e-5, 20
    train_args = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        base_ckpt=args.base_ckpt,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        beta=args.beta,
        clip_eps=args.clip_eps,
        learning_rate=lr,
        min_lr=min_lr,
        warmup_steps=warmup,
    )
    set_seed(train_args.seed)
    device = get_device()

    tokenizer = Tokenizer(train_args.tokenizer_name)
    config = NanoConfig()
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
    model = LlamaForCausalLM(config)
    if train_args.base_ckpt:
        load_base_weights(model, train_args.base_ckpt)
    # ref 必须在 inject_lora 之前拷贝，否则 KL 锚点也会带上适配器
    ref = build_ref_model(model) if args.grpo else None
    if train_args.lora:
        inject_lora(model, r=train_args.lora_r, alpha=train_args.lora_alpha)
        trainable, total = lora_param_stats(model)
        print(f"lora trainable {trainable}/{total} ({100 * trainable / total:.2f}%)", flush=True)
    model = model.to(device)
    if ref is not None:
        ref = ref.to(device)
    if args.grpo:
        max_prompt = max(train_args.seq_len - train_args.max_new_tokens, 1)
        dataset = GrpoJsonlDataset(train_args.data_path, tokenizer, max_prompt)
        dataloader = DataLoader(
            dataset,
            batch_size=train_args.micro_batch_size,
            collate_fn=lambda batch: grpo_collate(batch, tokenizer.pad_token_id),
            num_workers=train_args.num_workers,
        )
        GrpoTrainer(model, ref, tokenizer, dataloader, train_args, device).train()
        return
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
