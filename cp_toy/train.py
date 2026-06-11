from __future__ import annotations

import json
import random
from dataclasses import asdict
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
    model_weight_markers,
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


def query_marker_at_step(step: int, cfg: TrainConfig) -> str:
    if cfg.query_marker_after_intro is not None and cfg.intro_step is not None and step >= cfg.intro_step:
        return cfg.query_marker_after_intro
    return cfg.query_marker


def token_pool_at_step(step: int, cfg: TrainConfig) -> str:
    if cfg.token_pool_after_intro is not None and cfg.intro_step is not None and step >= cfg.intro_step:
        return cfg.token_pool_after_intro
    return cfg.token_pool


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


def should_reset_optimizer_state(step: int, sched_cfg: ScheduleConfig) -> bool:
    """Whether to reset AdamW moment state at this step for C5b reset control."""
    return (
        sched_cfg.kind == "warmup_cosine_then_rewarm_constant_reset_optim"
        and sched_cfg.rewarm_step is not None
        and step == sched_cfg.rewarm_step
    )


def reset_optimizer_state(opt: torch.optim.Optimizer) -> None:
    """Clear AdamW state while preserving parameters and param-group hyperparameters."""
    opt.state.clear()


def _add_prefixed(dst: Dict[str, float], prefix: str, src: Dict[str, float]) -> None:
    for k, v in src.items():
        dst[f"{prefix}{k}"] = v


def _evaluate_core_for(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    query_marker: str,
    token_pool: str,
    prefix: str = "",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    by_hop = evaluate_by_hop(
        model,
        gen,
        batch_size=train_cfg.batch_size,
        num_batches=train_cfg.eval_batches,
        device=device,
        k_max=data_cfg.k_max,
        query_marker=query_marker,
        token_pool=token_pool,
    )
    _add_prefixed(metrics, prefix, by_hop)
    # Content-shuffled floor tracked per checkpoint for hop1 and max hop.
    metrics[f"{prefix}floor_hop1_acc"] = content_shuffled_floor(
        model, gen, train_cfg.batch_size, max(1, train_cfg.eval_batches // 2), device, force_hop=1, token_pool=token_pool
    )
    metrics[f"{prefix}floor_hop{data_cfg.k_max}_acc"] = content_shuffled_floor(
        model,
        gen,
        train_cfg.batch_size,
        max(1, train_cfg.eval_batches // 2),
        device,
        force_hop=data_cfg.k_max,
        token_pool=token_pool,
    )
    scores = key_slot_lookup_scores(
        model, gen, batch_size=train_cfg.batch_size, num_batches=max(1, train_cfg.eval_batches // 2), device=device, token_pool=token_pool
    )
    metrics[f"{prefix}keyslot_top1"] = float(scores.max().item())
    metrics[f"{prefix}keyslot_mean"] = float(scores.mean().item())
    flat_idx = int(scores.flatten().argmax().item())
    metrics[f"{prefix}keyslot_top_layer"] = float(flat_idx // model.cfg.n_heads)
    metrics[f"{prefix}keyslot_top_head"] = float(flat_idx % model.cfg.n_heads)
    return metrics


def evaluate_core(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    current_query_marker: str,
    current_token_pool: str,
) -> Dict[str, float]:
    """Evaluate the main condition and optional specificity-control panels."""
    metrics = _evaluate_core_for(model, gen, data_cfg, train_cfg, device, current_query_marker, current_token_pool, prefix="")

    if train_cfg.eval_query_b:
        b = _evaluate_core_for(model, gen, data_cfg, train_cfg, device, "B", current_token_pool, prefix="queryB_")
        metrics.update(b)
        a = _evaluate_core_for(model, gen, data_cfg, train_cfg, device, "A", current_token_pool, prefix="queryA_")
        metrics.update(a)

    if train_cfg.eval_base_fresh:
        base = _evaluate_core_for(model, gen, data_cfg, train_cfg, device, current_query_marker, "base", prefix="base_")
        fresh = _evaluate_core_for(model, gen, data_cfg, train_cfg, device, current_query_marker, "fresh", prefix="fresh_")
        metrics.update(base)
        metrics.update(fresh)

    if train_cfg.log_rank_metrics:
        metrics.update(model_weight_markers(model))
    return metrics


def _param_norm_sq(model: TinyTransformer) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += float((p.detach().float() ** 2).sum().item())
    return total


def train_run(
    data_cfg: DataConfig,
    model_cfg: Optional[ModelConfig],
    optim_cfg: OptimConfig,
    sched_cfg: ScheduleConfig,
    train_cfg: TrainConfig,
) -> Path:
    """Run one training condition and write JSONL logs."""
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
            optimizer_reset = False
            if should_reset_optimizer_state(step, sched_cfg):
                reset_optimizer_state(opt)
                optimizer_reset = True
            lr = lr_at_step(step, optim_cfg, sched_cfg)
            set_optimizer_lr(opt, lr)
            p_dyn = p_dynamic_at_step(step, train_cfg)
            p_mul = p_multi_at_step(step, train_cfg)
            q_marker = query_marker_at_step(step, train_cfg)
            tok_pool = token_pool_at_step(step, train_cfg)
            batch = gen.batch(
                batch_size=train_cfg.batch_size,
                p_dynamic=p_dyn,
                p_multi=p_mul,
                query_marker=q_marker,
                token_pool=tok_pool,
                device=device,
            )
            out = model(batch.input_ids)
            loss = masked_ce_loss(out["logits"], batch.labels, batch.loss_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if optim_cfg.grad_clip and optim_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)

            before_sq = _param_norm_sq(model) if train_cfg.log_rank_metrics else None
            before_params = None
            if train_cfg.log_rank_metrics:
                before_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            opt.step()

            update_to_weight_ratio = None
            if train_cfg.log_rank_metrics and before_params is not None and before_sq is not None:
                delta_sq = 0.0
                idx = 0
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    delta = p.detach().float() - before_params[idx].float().to(p.device)
                    delta_sq += float((delta * delta).sum().item())
                    idx += 1
                after_sq = _param_norm_sq(model)
                update_to_weight_ratio = (delta_sq ** 0.5) / max(after_sq ** 0.5, 1e-12)

            if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1:
                row = {
                    "step": step,
                    "lr": lr,
                    "optimizer_reset": optimizer_reset,
                    "train_loss": float(loss.item()),
                    "train_acc": batch_accuracy(out["logits"].detach(), batch),
                    "p_dynamic": p_dyn,
                    "p_multi": p_mul,
                    "query_marker": q_marker,
                    "token_pool": tok_pool,
                }
                if update_to_weight_ratio is not None:
                    row["update_to_weight_ratio"] = float(update_to_weight_ratio)
                row.update(evaluate_core(model, gen, data_cfg, train_cfg, device, q_marker, tok_pool))
                log_f.write(json.dumps(row, sort_keys=True) + "\n")
                log_f.flush()

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    return out_dir
