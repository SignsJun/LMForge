import copy
import json
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .data import Tokenizer
from .model import LlamaForCausalLM
from .trainer import TrainConfig, Trainer


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # logits[t] 预测的是 input_ids[t+1]，取出实际采到的那个 token 的 logπ
    
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    # 同一 prompt 的 G 条回答互相当 baseline，(r-mean)/std 作为 advantage
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((grouped - mean) / std).reshape(-1)


def completion_mask(input_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    # logprobs 对齐 input_ids[:, 1:]，prompt 最后一个 token 才开始预测 completion
    targets = input_ids[:, 1:]
    pos = torch.arange(targets.size(1), device=input_ids.device)
    in_completion = pos >= (prompt_len - 1)
    # 第一个 eos 计入 loss，eos 之后的 token 全部 mask 掉
    eos = (targets == eos_id) & in_completion.unsqueeze(0)
    return (in_completion.unsqueeze(0) & (eos.int().cumsum(dim=-1) <= 1)).to(dtype=torch.float32)


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    # μ=1 时 old=logπ.detach()，ratio 恒为 1，clip 不生效，但公式按完整 PPO-clip 写
    ratio = (logprobs - old_logprobs).exp()
    adv = advantages.unsqueeze(-1)
    policy = -torch.min(ratio * adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv)
    # Schulman 无偏 KL：π_ref/π_θ - log(π_ref/π_θ) - 1
    kl = (ref_logprobs - logprobs).exp() - (ref_logprobs - logprobs) - 1.0
    loss = (policy + beta * kl) * mask
    return loss.sum() / mask.sum().clamp_min(1)


def rule_reward(
    tokenizer: Tokenizer,
    sequences: torch.Tensor,
    prompt_len: int,
    answers: list[str],
) -> torch.Tensor:
    # answer 不当 CE 标签，只检查生成文本里是否包含金标
    rewards = []
    for ids, answer in zip(sequences, answers):
        text = tokenizer.decode(ids[prompt_len:].tolist())
        rewards.append(1.0 if answer.strip() and answer.strip() in text else 0.0)
    return torch.tensor(rewards, device=sequences.device, dtype=torch.float32)


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_ref_model(model: LlamaForCausalLM) -> LlamaForCausalLM:
    # KL 锚点：注入 LoRA 之前的冻结拷贝，不参与梯度
    return freeze_model(copy.deepcopy(model))


def _prompt_text(obj: dict) -> tuple[str, str]:
    # 数据只提供 prompt 和用来打分的 answer，不提供要模仿的 completion
    answer = str(obj.get("answer") or obj.get("gold") or "")
    if obj.get("prompt"):
        return str(obj["prompt"]), answer
    conv = obj.get("conversations") or obj.get("messages") or []
    lines = []
    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant" and not answer:
            answer = content
    lines.append("Assistant:")
    return "\n".join(lines), answer


def _iter_grpo_jsonl(path: Path) -> Iterator[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


class GrpoJsonlDataset(IterableDataset):
    def __init__(self, path: str | Path, tokenizer: Tokenizer, max_prompt_len: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def __iter__(self):
        for obj in _iter_grpo_jsonl(self.path):
            text, answer = _prompt_text(obj)
            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            # 截断左侧，给后面 generate 留出 max_new_tokens
            ids = ids[-self.max_prompt_len :]
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "answer": answer,
            }


def grpo_collate(batch: list[dict], pad_id: int) -> dict:
    # 左 padding：解码时 pad 在因果注意力左侧，不会污染 prompt
    max_len = max(item["input_ids"].numel() for item in batch)
    ids, mask = [], []
    for item in batch:
        n = max_len - item["input_ids"].numel()
        ids.append(F.pad(item["input_ids"], (n, 0), value=pad_id))
        mask.append(F.pad(item["attention_mask"], (n, 0), value=0))
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(mask),
        "answers": [item["answer"] for item in batch],
    }


class GrpoTrainer(Trainer):
    def __init__(
        self,
        model: LlamaForCausalLM,
        ref: LlamaForCausalLM,
        tokenizer: Tokenizer,
        dataloader: DataLoader,
        args: TrainConfig,
        device: torch.device,
    ):
        super().__init__(model, dataloader, args, device)
        self.ref = ref
        self.tokenizer = tokenizer

    def train(self) -> None:
        args = self.args
        g = args.group_size
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_reward = 0.0
        t0 = time.time()
        accum = 0
        data = iter(self.dataloader)
        eos_id = self.tokenizer.eos_token_id

        while self.global_step < args.max_steps:
            try:
                batch = next(data)
            except StopIteration:
                data = iter(self.dataloader)
                batch = next(data)

            prompt_ids = batch["input_ids"].to(self.device, non_blocking=True)
            prompt_attn = batch["attention_mask"].to(self.device, non_blocking=True)
            # 每条 prompt 重复 G 次，采样一组回答再做组内相对优势，让1个答案多次使用
            answers = [a for a in batch["answers"] for _ in range(g)]
            prompt_ids = prompt_ids.repeat_interleave(g, dim=0)
            prompt_attn = prompt_attn.repeat_interleave(g, dim=0)
            prompt_len = prompt_ids.size(1)

            self.model.eval()
            with torch.no_grad():
                # 采样断开梯度；跑满 max_new_tokens，eos 只靠 completion_mask 丢掉
                sequences = self.model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    eos_token_id=eos_id,
                    attention_mask=prompt_attn,
                    stop_on_eos=False,
                ) # seq shape (B * G, L)
                comp = sequences.size(1) - prompt_len # comp shape (B * G, L - prompt_len)
                # 这里的 attn mask 是 padding mask
                attn = torch.cat(
                    [
                        prompt_attn,
                        torch.ones(prompt_ids.size(0), comp, device=self.device, dtype=prompt_attn.dtype),
                    ],
                    dim=1,
                )
                ref_logits, _, _ = self.ref(sequences, attention_mask=attn)

            # 对采到的 token 再 teacher-force 一遍，才能反传 logπ
            self.model.train()
            logits, _, _ = self.model(sequences, attention_mask=attn) #
            logp = token_logprobs(logits, sequences)
            refp = token_logprobs(ref_logits, sequences)
            mask = completion_mask(sequences, prompt_len, eos_id)
            rewards = rule_reward(self.tokenizer, sequences, prompt_len, answers)
            adv = group_advantages(rewards, g)
            loss = grpo_loss(
                logp,
                logp.detach(),
                refp,
                adv,
                mask,
                clip_eps=args.clip_eps,
                beta=args.beta,
            )
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()
            running_reward += rewards.mean().item() / args.gradient_accumulation_steps
            accum += 1
            if accum < args.gradient_accumulation_steps:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            accum = 0

            if self.global_step % args.logging_steps == 0:
                dt = time.time() - t0
                print(
                    f"step {self.global_step}/{args.max_steps}  "
                    f"loss {running_loss / args.logging_steps:.4f}  "
                    f"reward {running_reward / args.logging_steps:.3f}  "
                    f"lr {self.lr_scheduler.get_last_lr()[0]:.2e}  "
                    f"{dt:.1f}s",
                    flush=True,
                )
                running_loss = 0.0
                running_reward = 0.0
                t0 = time.time()

            if self.global_step % args.save_steps == 0:
                self._save_checkpoint(f"{args.output_dir}/checkpoint-{self.global_step}.pt")

        self._save_checkpoint(f"{args.output_dir}/pytorch_model.pt")
