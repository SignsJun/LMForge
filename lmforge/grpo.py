import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lmforge.data.tokenizer import Tokenizer
from lmforge.trainer.checkpoint import save_checkpoint
from lmforge.trainer.trainer import Trainer
from lmforge.utils.distributed import all_reduce_mean, is_main, unwrap


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((grouped - mean) / std).reshape(-1)


def completion_mask(input_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    targets = input_ids[:, 1:]
    pos = torch.arange(targets.size(1), device=input_ids.device)
    in_completion = pos >= (prompt_len - 1)
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
    ratio = (logprobs - old_logprobs).exp()
    adv = advantages.unsqueeze(-1)
    policy = -torch.min(ratio * adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv)
    kl = (ref_logprobs - logprobs).exp() - (ref_logprobs - logprobs) - 1.0
    loss = (policy + beta * kl) * mask
    return loss.sum() / mask.sum().clamp_min(1)


def rule_reward(
    tokenizer: Tokenizer,
    sequences: torch.Tensor,
    prompt_len: int,
    answers: list[str],
) -> torch.Tensor:
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


def build_ref_model(model: nn.Module) -> nn.Module:
    return freeze_model(copy.deepcopy(model))


class GrpoTrainer(Trainer):
    def __init__(self, model, ref, tokenizer, dataloader, config, device):
        super().__init__(model, dataloader, config, device)
        self.ref = ref
        self.tokenizer = tokenizer

    def train(self) -> None:
        cfg = self.config
        g = cfg.group_size
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_reward = 0.0
        t0 = time.time()
        accum = 0
        eos_id = self.tokenizer.eos_id
        policy = unwrap(self.model)

        for batch in self._batches():
            prompt_ids = batch["input_ids"].to(self.device, non_blocking=True)
            prompt_attn = batch["attention_mask"].to(self.device, non_blocking=True)
            answers = [a for a in batch["answers"] for _ in range(g)]
            prompt_ids = prompt_ids.repeat_interleave(g, dim=0)
            prompt_attn = prompt_attn.repeat_interleave(g, dim=0)
            prompt_len = prompt_ids.size(1)

            policy.eval()
            with torch.no_grad(), self.autocast:
                sequences = policy.generate(
                    prompt_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    eos_token_id=eos_id,
                    attention_mask=prompt_attn,
                    stop_on_eos=False,
                )
                comp = sequences.size(1) - prompt_len
                attn = torch.cat(
                    [
                        prompt_attn,
                        torch.ones(
                            prompt_ids.size(0),
                            comp,
                            device=self.device,
                            dtype=prompt_attn.dtype,
                        ),
                    ],
                    dim=1,
                )
                ref_logits, _ = self.ref(sequences, attention_mask=attn)

            self.model.train()
            with self.autocast:
                logits, _ = self.model(sequences, attention_mask=attn)
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
                    clip_eps=cfg.clip_eps,
                    beta=cfg.beta,
                )
                loss = loss / cfg.grad_accum_steps

            sync = accum + 1 >= cfg.grad_accum_steps
            if sync or not hasattr(self.model, "no_sync"):
                loss.backward()
            else:
                with self.model.no_sync():
                    loss.backward()
            running_loss += loss.item()
            running_reward += rewards.mean().item() / cfg.grad_accum_steps
            accum += 1
            if not sync:
                continue

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            accum = 0

            if self.step % cfg.log_interval == 0:
                dt = time.time() - t0
                avg_loss = all_reduce_mean(running_loss / cfg.log_interval, self.device)
                avg_reward = all_reduce_mean(running_reward / cfg.log_interval, self.device)
                if is_main():
                    print(
                        f"step {self.step}/{cfg.max_steps}  "
                        f"loss {avg_loss:.4f}  "
                        f"reward {avg_reward:.3f}  "
                        f"lr {self.scheduler.get_last_lr()[0]:.2e}  "
                        f"{dt:.1f}s",
                        flush=True,
                    )
                running_loss = 0.0
                running_reward = 0.0
                t0 = time.time()

            if self.step % cfg.save_interval == 0:
                save_checkpoint(
                    f"{cfg.output_dir}/step_{self.step}.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                    extra=self._ckpt_extra(),
                )
            if self.step >= cfg.max_steps:
                break

        save_checkpoint(
            f"{cfg.output_dir}/last.pt",
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            extra=self._ckpt_extra(),
        )
