#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cp_toy.config import DataConfig, ModelConfig, TrainConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import evaluate_by_hop, key_slot_lookup_scores
from cp_toy.model import TinyTransformer


def _filter_dataclass(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def load_run_config(run_dir: Path) -> Tuple[DataConfig, ModelConfig, TrainConfig, Dict[str, Any]]:
    cfg = json.loads((run_dir / "config.json").read_text())
    data = DataConfig(**_filter_dataclass(DataConfig, cfg.get("data", {})))
    model_cfg = ModelConfig(**_filter_dataclass(ModelConfig, cfg.get("model", {})))
    train = TrainConfig(**_filter_dataclass(TrainConfig, cfg.get("train", {})))
    return data, model_cfg, train, cfg


def find_run_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("config.json"):
        d = p.parent
        if (d / "checkpoint_pre_intro.pt").exists() and (d / "model_final.pt").exists():
            out.append(d)
    return sorted(out)


def run_label(run_dir: Path, root: Path) -> str:
    try:
        rel = run_dir.relative_to(root)
    except ValueError:
        rel = run_dir
    parts = rel.parts
    # Prefer the parent arm directory if structure is seedX/arm/late_gate_post_seedY.
    if len(parts) >= 2 and parts[-1].startswith("late_gate"):
        return "/".join(parts[:-1])
    return str(rel)


def load_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location=device)


def make_model(model_cfg: ModelConfig, state: Dict[str, torch.Tensor], device: torch.device) -> TinyTransformer:
    model = TinyTransformer(model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def float_keys(state: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k, v in state.items() if torch.is_floating_point(v)]


def state_to_vector(state: Dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    return torch.cat([state[k].detach().float().flatten().cpu() for k in keys])


def vector_to_state(template: Dict[str, torch.Tensor], keys: Sequence[str], vec: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in template.items()}
    offset = 0
    for k in keys:
        v = template[k]
        n = v.numel()
        chunk = vec[offset : offset + n].reshape(v.shape).to(device=v.device, dtype=v.dtype)
        out[k] = chunk
        offset += n
    if offset != vec.numel():
        raise ValueError(f"vector length mismatch: consumed {offset}, vector has {vec.numel()}")
    return out


def interpolate_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None and torch.is_floating_point(va):
            out[k] = (1.0 - alpha) * va + alpha * vb
        else:
            out[k] = va.detach().clone()
    return out


def norm(v: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(v).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(den.item()) <= 1e-20:
        return float("nan")
    return float(torch.dot(a, b).item() / den.item())


@torch.no_grad()
def eval_state(
    state: Dict[str, torch.Tensor],
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    batch_size: int,
    eval_batches: int,
    seed: int,
    token_pool: str,
) -> Dict[str, float]:
    model = make_model(model_cfg, state, device)
    gen = ChainBatchGenerator(data_cfg, seed=seed)
    ev = evaluate_by_hop(
        model,
        gen,
        batch_size=batch_size,
        num_batches=eval_batches,
        device=device,
        k_max=data_cfg.k_max,
        query_marker="A",
        token_pool=token_pool,
    )
    try:
        scores = key_slot_lookup_scores(
            model,
            gen,
            batch_size=batch_size,
            num_batches=max(1, eval_batches // 2),
            device=device,
            token_pool=token_pool,
        )
        ev["keyslot_top1"] = float(scores.max().item())
        ev["keyslot_mean"] = float(scores.mean().item())
    except Exception:
        pass
    return ev


def metric(ev: Dict[str, Any], key: str) -> Optional[float]:
    x = ev.get(key)
    return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else None


def read_tail_from_summary(run_dir: Path, key: str) -> Optional[float]:
    sp = run_dir / "summary.json"
    if not sp.exists():
        return None
    try:
        s = json.loads(sp.read_text())
        # Try common nested keys.
        for container in (s.get("tail", {}), s.get("tail_metrics", {}), s.get("final", {}), s.get("final_metrics", {})):
            if isinstance(container, dict) and isinstance(container.get(key), (int, float)):
                return float(container[key])
    except Exception:
        return None
    return None


def parse_alphas(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def checkpoint_sort_key(p: Path) -> Tuple[int, str]:
    if p.name == "checkpoint_pre_intro.pt":
        return (-1, p.name)
    m = re.search(r"checkpoint_pre_step_(\d+)\.pt", p.name)
    if m:
        return (int(m.group(1)), p.name)
    if p.name == "model_final.pt":
        return (10**12, p.name)
    return (10**11, p.name)


def checkpoint_label(p: Path) -> str:
    if p.name == "checkpoint_pre_intro.pt":
        return "intro"
    m = re.search(r"checkpoint_pre_step_(\d+)\.pt", p.name)
    if m:
        return m.group(1)
    if p.name == "model_final.pt":
        return "final"
    return p.stem


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Path/subspace reachability probes for checkpointed toy runs")
    ap.add_argument("--runs-dir", required=True, help="Root containing checkpointed run dirs")
    ap.add_argument("--target-substring", default="rewarm_reset", help="Run label substring used as successful target direction")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--token-pool", default="all", choices=["all", "base", "fresh"])
    ap.add_argument("--own-alphas", default="0,0.25,0.5,0.65,0.75,0.85,0.9,0.95,1.0")
    ap.add_argument("--target-direction-alphas", default="0,0.25,0.5,0.75,1.0,1.25")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "reachability_geometry"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    run_dirs = find_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"no checkpointed run dirs found under {root}")

    labels = {d: run_label(d, root) for d in run_dirs}
    target_candidates = [d for d in run_dirs if args.target_substring in labels[d]]
    if not target_candidates:
        raise FileNotFoundError(f"no target run matching substring {args.target_substring!r}; labels={list(labels.values())}")
    # If multiple, prefer one with highest final hop2 from summary; otherwise first.
    target_run = max(target_candidates, key=lambda d: (read_tail_from_summary(d, "hop2_acc") or -1.0))
    target_label = labels[target_run]

    target_data, target_model_cfg, target_train, _ = load_run_config(target_run)
    target_intro = load_state(target_run / "checkpoint_pre_intro.pt", device)
    target_final = load_state(target_run / "model_final.pt", device)
    keys = float_keys(target_intro)
    theta_target_intro = state_to_vector(target_intro, keys)
    theta_target_final = state_to_vector(target_final, keys)
    target_delta = theta_target_final - theta_target_intro
    target_delta_norm = norm(target_delta)
    target_unit = target_delta / max(target_delta_norm, 1e-20)

    direction_rows: List[Dict[str, Any]] = []
    own_interp_rows: List[Dict[str, Any]] = []
    target_injection_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []

    own_alphas = parse_alphas(args.own_alphas)
    target_alphas = parse_alphas(args.target_direction_alphas)

    for idx, run_dir in enumerate(run_dirs):
        label = labels[run_dir]
        data_cfg, model_cfg, train_cfg, _ = load_run_config(run_dir)
        intro_state = load_state(run_dir / "checkpoint_pre_intro.pt", device)
        final_state = load_state(run_dir / "model_final.pt", device)
        # Require same tensor layout.
        run_keys = float_keys(intro_state)
        if run_keys != keys:
            print(f"warning: skipping {label}: state layout differs from target", file=sys.stderr)
            continue
        theta_intro = state_to_vector(intro_state, keys)
        theta_final = state_to_vector(final_state, keys)
        delta = theta_final - theta_intro
        theta_norm = norm(theta_intro)
        delta_norm = norm(delta)
        cos_to_target = cosine(delta, target_delta)
        proj_len = float(torch.dot(delta, target_unit).item())
        proj_frac_of_target = proj_len / max(target_delta_norm, 1e-20)
        residual = delta - proj_len * target_unit
        residual_norm = norm(residual)
        frac_move_in_target_dir = (proj_len * proj_len) / max(delta_norm * delta_norm, 1e-20)

        eval_intro = eval_state(intro_state, data_cfg, model_cfg, train_cfg, device, args.batch_size, args.eval_batches, seed=int(train_cfg.seed) + 11000 + idx, token_pool=args.token_pool)
        eval_final = eval_state(final_state, data_cfg, model_cfg, train_cfg, device, args.batch_size, args.eval_batches, seed=int(train_cfg.seed) + 12000 + idx, token_pool=args.token_pool)

        direction_rows.append({
            "run_label": label,
            "is_target_run": int(run_dir == target_run),
            "target_label": target_label,
            "seed": int(train_cfg.seed),
            "relative_distance_intro_to_final": delta_norm / max(theta_norm, 1e-20),
            "absolute_distance_intro_to_final": delta_norm,
            "cosine_delta_with_target_success_delta": cos_to_target,
            "projection_length_on_target_unit": proj_len,
            "projection_fraction_of_target_delta": proj_frac_of_target,
            "residual_norm_after_target_projection": residual_norm,
            "fraction_delta_energy_in_target_direction": frac_move_in_target_dir,
            "intro_hop1_acc": metric(eval_intro, "hop1_acc"),
            "intro_hop2_acc": metric(eval_intro, "hop2_acc"),
            "intro_hop2_loss": metric(eval_intro, "hop2_loss"),
            "final_hop1_acc": metric(eval_final, "hop1_acc"),
            "final_hop2_acc": metric(eval_final, "hop2_acc"),
            "final_hop2_loss": metric(eval_final, "hop2_loss"),
        })

        # Own straight-line interpolation intro -> final.
        for alpha in own_alphas:
            st = interpolate_state(intro_state, final_state, alpha)
            ev = eval_state(st, data_cfg, model_cfg, train_cfg, device, args.batch_size, args.eval_batches, seed=int(train_cfg.seed) + 13000 + int(alpha*1000) + idx, token_pool=args.token_pool)
            own_interp_rows.append({
                "run_label": label,
                "seed": int(train_cfg.seed),
                "alpha": alpha,
                "path": "own_intro_to_final",
                **ev,
            })

        # Add successful target delta to each run's intro state.
        for alpha in target_alphas:
            injected_vec = theta_intro + alpha * target_delta
            st = vector_to_state(intro_state, keys, injected_vec)
            ev = eval_state(st, data_cfg, model_cfg, train_cfg, device, args.batch_size, args.eval_batches, seed=int(train_cfg.seed) + 14000 + int(alpha*1000) + idx, token_pool=args.token_pool)
            target_injection_rows.append({
                "run_label": label,
                "seed": int(train_cfg.seed),
                "alpha": alpha,
                "target_label": target_label,
                "injected_delta": "target_success_delta",
                **ev,
            })

        # Saved checkpoint path segments and alignment to target direction.
        ckpts = sorted(
            [p for p in run_dir.glob("checkpoint_pre_intro.pt")] +
            [p for p in run_dir.glob("checkpoint_pre_step_*.pt")] +
            [run_dir / "model_final.pt"],
            key=checkpoint_sort_key,
        )
        prev_vec: Optional[torch.Tensor] = None
        prev_label: Optional[str] = None
        for j, ckpt in enumerate(ckpts):
            if not ckpt.exists():
                continue
            st = load_state(ckpt, device)
            vec = state_to_vector(st, keys)
            ev = eval_state(st, data_cfg, model_cfg, train_cfg, device, args.batch_size, args.eval_batches, seed=int(train_cfg.seed) + 15000 + j + idx, token_pool=args.token_pool)
            row: Dict[str, Any] = {
                "run_label": label,
                "seed": int(train_cfg.seed),
                "checkpoint": checkpoint_label(ckpt),
                "checkpoint_file": ckpt.name,
                "distance_from_intro_relative": norm(vec - theta_intro) / max(theta_norm, 1e-20),
                "projection_from_intro_on_target_delta_fraction": float(torch.dot(vec - theta_intro, target_unit).item()) / max(target_delta_norm, 1e-20),
                **ev,
            }
            if prev_vec is not None:
                seg = vec - prev_vec
                row.update({
                    "previous_checkpoint": prev_label,
                    "segment_norm": norm(seg),
                    "segment_cosine_with_target_success_delta": cosine(seg, target_delta),
                    "segment_projection_on_target_unit": float(torch.dot(seg, target_unit).item()),
                })
            checkpoint_rows.append(row)
            prev_vec = vec
            prev_label = checkpoint_label(ckpt)

    write_csv(out_dir / "directional_summary.csv", direction_rows)
    write_csv(out_dir / "own_path_interpolation.csv", own_interp_rows)
    write_csv(out_dir / "target_direction_injection.csv", target_injection_rows)
    write_csv(out_dir / "checkpoint_path_alignment.csv", checkpoint_rows)
    meta = {
        "runs_dir": str(root),
        "out_dir": str(out_dir),
        "target_run": str(target_run),
        "target_label": target_label,
        "target_delta_norm": target_delta_norm,
        "device": str(device),
        "interpretation": {
            "directional_summary": "Compares each arm's actual intro-to-final displacement with the successful target displacement.",
            "target_direction_injection": "Tests whether applying the successful displacement direction to each arm's own intro checkpoint is sufficient for HOP_2.",
            "checkpoint_path_alignment": "Measures how saved training segments align with the successful target direction and when HOP_2 appears along the path."
        }
    }
    (out_dir / "reachability_geometry_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"target_label": target_label, "target_delta_norm": target_delta_norm, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
