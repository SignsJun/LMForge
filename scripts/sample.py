import argparse

import torch

from lmforge.data import Tokenizer
from lmforge.lora import has_lora_state, inject_lora
from lmforge.models import LlamaForCausalLM, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/tiny.yaml")
    parser.add_argument("--ckpt", default="outputs/mac/last.pt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = LlamaForCausalLM(ModelConfig.from_yaml(args.model))
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
    tokenizer = Tokenizer(args.tokenizer)
    ids = tokenizer.encode(args.prompt)
    x = torch.tensor([ids if ids else [tokenizer.eos_id]], device=device)
    out = model.generate(x, args.max_new_tokens, args.temperature, tokenizer.eos_id)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
