import argparse

import torch

from .config import NanoConfig
from .data import Tokenizer
from .lora import inject_lora, has_lora_state
from .model import LlamaForCausalLM
from .trainer import get_device


def build_prompt(messages: list[tuple[str, str]]) -> str:
    lines = [f"{role}: {content}" for role, content in messages]
    lines.append("Assistant:")
    return "\n".join(lines)


def reply(model, tokenizer, messages, max_new_tokens, temperature, device) -> str:
    prompt = build_prompt(messages)
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    if input_ids.numel() == 0:
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0].tolist())
    if text.startswith(prompt):
        text = text[len(prompt):]
    text = text.split("\nUser:")[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/nano/pytorch_model.pt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = get_device()
    tokenizer = Tokenizer(args.tokenizer)
    config = NanoConfig()
    config.vocab_size = max(config.vocab_size, tokenizer.vocab_size)
    model = LlamaForCausalLM(config)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    lora_cfg = ckpt.get("lora")
    if lora_cfg or has_lora_state(state):
        r = lora_cfg["r"] if lora_cfg else 8
        alpha = lora_cfg["alpha"] if lora_cfg else 16.0
        inject_lora(model, r=r, alpha=alpha)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    messages: list[tuple[str, str]] = []
    if args.prompt:
        messages.append(("User", args.prompt))
        text = reply(model, tokenizer, messages, args.max_new_tokens, args.temperature, device)
        messages.append(("Assistant", text))
        print(f"Assistant: {text}")

    print("User: ", end="", flush=True)
    while True:
        try:
            user = input().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit", "q"}:
            break
        if not user:
            print("User: ", end="", flush=True)
            continue
        messages.append(("User", user))
        text = reply(model, tokenizer, messages, args.max_new_tokens, args.temperature, device)
        messages.append(("Assistant", text))
        print(f"Assistant: {text}")
        print("User: ", end="", flush=True)


if __name__ == "__main__":
    main()
