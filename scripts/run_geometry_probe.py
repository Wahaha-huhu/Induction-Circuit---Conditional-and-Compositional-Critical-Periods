#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cp_toy.config import DataConfig, ModelConfig, OptimConfig, ScheduleConfig, TrainConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import batch_accuracy, evaluate_by_hop, key_slot_lookup_scores, masked_ce_loss
from cp_toy.model import TinyTransformer


def _filter_dataclass(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def load_run_config(run_dir: Path) -> Tuple[DataConfig, ModelConfig, TrainConfig]:
    cfg = json.loads((run_dir / "config.json").read_text())
    data = DataConfig(**_filter_dataclass(DataConfig, cfg.get("data", {})))
    model_cfg = ModelConfig(**_filter_dataclass(ModelConfig, cfg.get("model", {})))
    train = TrainConfig(**_filter_dataclass(TrainConfig, cfg.get("train", {})))
    return data, model_cfg, train


def load_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location=device)


def param_vector_from_model(model: TinyTransformer) -> torch.Tensor:
    return torch.cat([p.detach().float().flatten().cpu() for p in model.parameters() if p.requires_grad])


def grad_vector_from_model(model: TinyTransformer) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            parts.append(torch.zeros_like(p.detach()).float().flatten().cpu())
        else:
            parts.append(p.grad.detach().float().flatten().cpu())
    return torch.cat(parts)


def vector_norm(v: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(v).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 1e-20:
        return float("nan")
    return float(torch.dot(a, b).item() / denom.item())


def make_model(model_cfg: ModelConfig, state: Dict[str, torch.Tensor], device: torch.device) -> TinyTransformer:
    model = TinyTransformer(model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def hop2_gradient(
    model: TinyTransformer,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    batch_size: int,
    seed: int,
    token_pool: str = "all",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    gen = ChainBatchGenerator(data_cfg, seed=seed)
    batch = gen.batch(
        batch_size=batch_size,
        p_dynamic=1.0,
        p_multi=0.0,
        force_dynamic=True,
        force_hop=min(2, data_cfg.k_max),
        query_marker="A",
        token_pool=token_pool,
        device=device,
    )
    model.train(False)
    model.zero_grad(set_to_none=True)
    out = model(batch.input_ids)
    loss = masked_ce_loss(out["logits"], batch.labels, batch.loss_mask)
    loss.backward()
    g = grad_vector_from_model(model)
    metrics = {"hop2_grad_loss": float(loss.item()), "hop2_grad_acc": batch_accuracy(out["logits"].detach(), batch)}
    return g, metrics


@torch.no_grad()
def eval_model(
    model: TinyTransformer,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    eval_batches: int,
    batch_size: int,
    seed: int,
    token_pool: str = "all",
) -> Dict[str, float]:
    gen = ChainBatchGenerator(data_cfg, seed=seed)
    return evaluate_by_hop(
        model,
        gen,
        batch_size=batch_size,
        num_batches=eval_batches,
        device=device,
        k_max=data_cfg.k_max,
        query_marker="A",
        token_pool=token_pool,
    )


def interpolate_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in a:
        if k in b and torch.is_floating_point(a[k]):
            out[k] = (1.0 - alpha) * a[k] + alpha * b[k]
        else:
            out[k] = a[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe parameter-space distance/alignment between an intro checkpoint and a later checkpoint")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint-a", default="checkpoint_pre_intro.pt", help="Starting checkpoint, relative to run dir unless absolute")
    ap.add_argument("--checkpoint-b", default="model_final.pt", help="Target checkpoint, relative to run dir unless absolute")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--grad-batch-size", type=int, default=512)
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,0.9,1.0")
    ap.add_argument("--token-pool", default="all", choices=["all", "base", "fresh"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_cfg, model_cfg, train_cfg = load_run_config(run_dir)
    ckpt_a = Path(args.checkpoint_a)
    ckpt_b = Path(args.checkpoint_b)
    if not ckpt_a.is_absolute():
        ckpt_a = run_dir / ckpt_a
    if not ckpt_b.is_absolute():
        ckpt_b = run_dir / ckpt_b
    if not ckpt_a.exists():
        raise FileNotFoundError(f"missing checkpoint-a: {ckpt_a}")
    if not ckpt_b.exists():
        raise FileNotFoundError(f"missing checkpoint-b: {ckpt_b}")

    state_a = load_state(ckpt_a, device)
    state_b = load_state(ckpt_b, device)
    model_a = make_model(model_cfg, state_a, device)
    model_b = make_model(model_cfg, state_b, device)
    theta_a = param_vector_from_model(model_a)
    theta_b = param_vector_from_model(model_b)
    update = theta_b - theta_a
    theta_norm = vector_norm(theta_a)
    update_norm = vector_norm(update)

    grad, grad_metrics = hop2_gradient(
        model_a,
        data_cfg,
        train_cfg,
        device,
        batch_size=args.grad_batch_size,
        seed=int(train_cfg.seed) + 4242,
        token_pool=args.token_pool,
    )
    descent = -grad
    grad_norm = vector_norm(grad)
    align = cosine(update, descent)
    projection_on_descent = float(torch.dot(update, descent / max(vector_norm(descent), 1e-20)).item()) if grad_norm > 1e-20 else float("nan")

    eval_a = eval_model(model_a, data_cfg, train_cfg, device, args.eval_batches, args.batch_size, int(train_cfg.seed) + 5000, args.token_pool)
    eval_b = eval_model(model_b, data_cfg, train_cfg, device, args.eval_batches, args.batch_size, int(train_cfg.seed) + 6000, args.token_pool)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    interp = []
    for alpha in alphas:
        st = interpolate_state(state_a, state_b, alpha)
        m = make_model(model_cfg, st, device)
        ev = eval_model(m, data_cfg, train_cfg, device, args.eval_batches, args.batch_size, int(train_cfg.seed) + 7000 + int(alpha * 1000), args.token_pool)
        row = {"alpha": alpha, **ev}
        try:
            gen = ChainBatchGenerator(data_cfg, seed=int(train_cfg.seed) + 8000 + int(alpha * 1000))
            scores = key_slot_lookup_scores(m, gen, batch_size=args.batch_size, num_batches=max(1, args.eval_batches // 2), device=device, token_pool=args.token_pool)
            row["keyslot_top1"] = float(scores.max().item())
            row["keyslot_mean"] = float(scores.mean().item())
        except Exception:
            pass
        interp.append(row)

    result = {
        "run_dir": str(run_dir),
        "checkpoint_a": str(ckpt_a),
        "checkpoint_b": str(ckpt_b),
        "seed": train_cfg.seed,
        "intro_step": train_cfg.intro_step,
        "theta_a_norm": theta_norm,
        "update_norm_a_to_b": update_norm,
        "relative_update_norm": update_norm / max(theta_norm, 1e-20),
        "hop2_grad_norm_at_a": grad_norm,
        "cosine_update_with_negative_hop2_grad_at_a": align,
        "projection_update_on_negative_hop2_grad_unit": projection_on_descent,
        **grad_metrics,
        "eval_a": eval_a,
        "eval_b": eval_b,
        "interpolation": interp,
        "interpretation_notes": {
            "cosine_update_with_negative_hop2_grad_at_a": "Positive means the eventual parameter displacement initially aligns with steepest HOP_2 loss descent at checkpoint A.",
            "relative_update_norm": "Approximate normalized distance travelled from conditioned starting checkpoint A to target checkpoint B.",
            "interpolation": "If HOP_2 appears abruptly over alpha, the path crosses a sharp behavioural/circuit threshold."
        }
    }

    out_path = Path(args.out) if args.out else run_dir / "geometry_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["run_dir", "relative_update_norm", "hop2_grad_norm_at_a", "cosine_update_with_negative_hop2_grad_at_a", "projection_update_on_negative_hop2_grad_unit"]}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
