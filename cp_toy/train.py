from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import DataConfig, ModelConfig, OptimConfig, ScheduleConfig, TrainConfig
from .data import ChainBatchGenerator
from .metrics import (
    batch_accuracy,
    content_shuffled_floor,
    evaluate_by_hop,
    key_slot_lookup_scores,
    masked_ce_loss,
)
from .model import TinyTransformer, parameter_count
from .schedules import lr_at_step


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def p_dynamic_at_step(step: int, cfg: TrainConfig) -> float:
    if cfg.dynamic_switch_step is None:
        return cfg.p_dynamic_high
    return cfg.p_dynamic_low if step < cfg.dynamic_switch_step else cfg.p_dynamic_high


def p_multi_at_step(step: int, cfg: TrainConfig) -> float:
    if cfg.intro_step is None:
        return cfg.p_multi_frozen
    return cfg.p_multi_before_intro if step < cfg.intro_step else cfg.p_multi_frozen


def make_optimizer(model: TinyTransformer, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.peak_lr,
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        weight_decay=optim_cfg.weight_decay,
    )


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def evaluate_core(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    by_hop = evaluate_by_hop(
        model,
        gen,
        batch_size=train_cfg.batch_size,
        num_batches=train_cfg.eval_batches,
        device=device,
        k_max=data_cfg.k_max,
        query_marker=train_cfg.query_marker,
    )
    metrics.update(by_hop)
    # Content-shuffled floor tracked per checkpoint for hop1 and max hop.
    metrics["floor_hop1_acc"] = content_shuffled_floor(
        model, gen, train_cfg.batch_size, max(1, train_cfg.eval_batches // 2), device, force_hop=1
    )
    metrics[f"floor_hop{data_cfg.k_max}_acc"] = content_shuffled_floor(
        model, gen, train_cfg.batch_size, max(1, train_cfg.eval_batches // 2), device, force_hop=data_cfg.k_max
    )
    scores = key_slot_lookup_scores(
        model, gen, batch_size=train_cfg.batch_size, num_batches=max(1, train_cfg.eval_batches // 2), device=device
    )
    metrics["keyslot_top1"] = float(scores.max().item())
    metrics["keyslot_mean"] = float(scores.mean().item())
    # Store top head coordinates compactly.
    flat_idx = int(scores.flatten().argmax().item())
    metrics["keyslot_top_layer"] = float(flat_idx // data_cfg.k_max if False else flat_idx // model.cfg.n_heads)
    metrics["keyslot_top_head"] = float(flat_idx % model.cfg.n_heads)
    return metrics


def train_run(
    data_cfg: DataConfig,
    model_cfg: Optional[ModelConfig],
    optim_cfg: OptimConfig,
    sched_cfg: ScheduleConfig,
    train_cfg: TrainConfig,
) -> Path:
    """Run one training condition and write JSONL logs.

    This is the base runner used by scripts/run_condition.py. It is intentionally
    condition-agnostic: intro_step and dynamic_switch_step implement C4/gate and
    C1-style manipulations without changing the training loop.
    """
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)

    if model_cfg is None:
        model_cfg = ModelConfig(vocab_size=data_cfg.vocab_size, seq_len=data_cfg.input_seq_len)
    model = TinyTransformer(model_cfg).to(device)
    opt = make_optimizer(model, optim_cfg)
    gen = ChainBatchGenerator(data_cfg, seed=train_cfg.seed + 1000)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": asdict(data_cfg),
                "model": asdict(model_cfg),
                "optim": asdict(optim_cfg),
                "schedule": asdict(sched_cfg),
                "train": asdict(train_cfg),
                "parameter_count": parameter_count(model),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    log_path = out_dir / "metrics.jsonl"
    with open(log_path, "w", encoding="utf-8") as log_f:
        for step in range(train_cfg.max_steps):
            model.train()
            lr = lr_at_step(step, optim_cfg, sched_cfg)
            set_optimizer_lr(opt, lr)
            p_dyn = p_dynamic_at_step(step, train_cfg)
            p_mul = p_multi_at_step(step, train_cfg)
            batch = gen.batch(
                batch_size=train_cfg.batch_size,
                p_dynamic=p_dyn,
                p_multi=p_mul,
                query_marker=train_cfg.query_marker,
                device=device,
            )
            out = model(batch.input_ids)
            loss = masked_ce_loss(out["logits"], batch.labels, batch.loss_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if optim_cfg.grad_clip and optim_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)
            opt.step()

            if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
                row = {
                    "step": step,
                    "lr": lr,
                    "train_loss": float(loss.item()),
                    "train_acc": batch_accuracy(out["logits"].detach(), batch),
                    "p_dynamic": p_dyn,
                    "p_multi": p_mul,
                }
                row.update(evaluate_core(model, gen, data_cfg, train_cfg, device))
                log_f.write(json.dumps(row, sort_keys=True) + "\n")
                log_f.flush()

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    return out_dir
