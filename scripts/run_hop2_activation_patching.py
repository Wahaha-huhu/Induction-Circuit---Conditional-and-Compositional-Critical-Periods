#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cp_toy.config import DataConfig, ModelConfig
from cp_toy.model import TinyTransformer
from cp_toy.train import resolve_device


def load_model(run_dir: Path, device: torch.device) -> Tuple[TinyTransformer, DataConfig, ModelConfig, Dict[str, Any]]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg["data"])
    model_cfg = ModelConfig(**cfg["model"])
    model = TinyTransformer(model_cfg).to(device)
    state = torch.load(run_dir / "model_final.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, data_cfg, model_cfg, cfg


def make_paired_hop2_batch(
    cfg: DataConfig,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Make clean/corrupt HOP_2 pairs with same first edge but changed second edge.

    Clean has A->B->C and target C.  Corrupt has A->B->D and target D.
    The query is HOP_2 A in both cases.  This lets activation patching ask whether
    clean activations restore the clean answer on the corrupt input.
    """
    clean_inputs: List[List[int]] = []
    corrupt_inputs: List[List[int]] = []
    clean_targets: List[int] = []
    corrupt_targets: List[int] = []
    s_positions: List[int] = []

    content = np.arange(cfg.v_content, dtype=np.int64)
    if cfg.chain_length + 1 > cfg.v_content:
        raise ValueError("need at least chain_length+1 content tokens for corrupt replacement")

    for _ in range(batch_size):
        # We need a valid HOP_2 start with s+2 in range.
        chain = rng.choice(content, size=cfg.chain_length, replace=False).astype(np.int64)
        remaining = np.setdiff1d(content, chain, assume_unique=False)
        replacement = int(rng.choice(remaining, size=1)[0])
        s = int(rng.integers(0, cfg.chain_length - 2))
        clean_chain = chain.copy()
        corrupt_chain = chain.copy()
        corrupt_chain[s + 2] = replacement

        binding_indices = list(range(cfg.chain_length - 1))
        rng.shuffle(binding_indices)

        def build_input(c: np.ndarray) -> List[int]:
            seq: List[int] = []
            for i in binding_indices:
                seq.extend([int(c[i]), int(c[i + 1]), int(cfg.sep_token)])
            seq.extend([int(cfg.query_a_token), int(cfg.hop_token_offset + 2), int(c[s])])
            return seq

        clean_inputs.append(build_input(clean_chain))
        corrupt_inputs.append(build_input(corrupt_chain))
        clean_targets.append(int(clean_chain[s + 2]))
        corrupt_targets.append(int(corrupt_chain[s + 2]))
        s_positions.append(s)

    return {
        "clean_input_ids": torch.tensor(clean_inputs, dtype=torch.long, device=device),
        "corrupt_input_ids": torch.tensor(corrupt_inputs, dtype=torch.long, device=device),
        "clean_target": torch.tensor(clean_targets, dtype=torch.long, device=device),
        "corrupt_target": torch.tensor(corrupt_targets, dtype=torch.long, device=device),
        "query_pos": torch.full((batch_size,), int(cfg.query_start_input_pos), dtype=torch.long, device=device),
    }


@torch.no_grad()
def logit_diff(logits: torch.Tensor, clean_target: torch.Tensor, corrupt_target: torch.Tensor) -> torch.Tensor:
    last = logits[:, -1, :]
    b = torch.arange(last.shape[0], device=last.device)
    return last[b, clean_target] - last[b, corrupt_target]


def summarize_vals(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "stderr": float("nan")}
    x = torch.tensor(vals, dtype=torch.float64)
    mean = float(x.mean().item())
    stderr = float((x.std(unbiased=False) / math.sqrt(max(1, x.numel()))).item())
    return {"mean": mean, "stderr": stderr}


@torch.no_grad()
def eval_patch_site(
    model: TinyTransformer,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    rng: np.random.Generator,
    batch_size: int,
    num_batches: int,
    patch_kind: str,
    layer: int | None = None,
    head: int | None = None,
) -> Dict[str, float]:
    clean_diffs: List[float] = []
    corrupt_diffs: List[float] = []
    patched_diffs: List[float] = []
    restorations: List[float] = []
    clean_accs: List[float] = []
    corrupt_accs: List[float] = []
    patched_clean_accs: List[float] = []

    for _ in range(num_batches):
        batch = make_paired_hop2_batch(data_cfg, batch_size, rng, next(model.parameters()).device)
        clean_ids = batch["clean_input_ids"]
        corrupt_ids = batch["corrupt_input_ids"]
        clean_target = batch["clean_target"]
        corrupt_target = batch["corrupt_target"]
        qpos = int(data_cfg.query_start_input_pos)
        b_idx = torch.arange(batch_size, device=clean_ids.device)

        clean_out = model(clean_ids, return_residuals=True, return_head_outputs=True)
        corrupt_out = model(corrupt_ids)
        c_diff = logit_diff(clean_out["logits"], clean_target, corrupt_target)
        z_diff = logit_diff(corrupt_out["logits"], clean_target, corrupt_target)

        if patch_kind == "residual":
            assert layer is not None
            patched_out = model(
                corrupt_ids,
                patch_residuals={layer: {"source": clean_out["residuals"][layer], "positions": qpos}},
            )
        elif patch_kind == "head":
            assert layer is not None and head is not None
            patched_out = model(
                corrupt_ids,
                patch_head_outputs={layer: {"source": clean_out["head_outputs"][layer], "heads": [head], "positions": qpos}},
            )
        elif patch_kind == "none":
            patched_out = corrupt_out
        else:
            raise ValueError(f"unknown patch_kind {patch_kind}")

        p_diff = logit_diff(patched_out["logits"], clean_target, corrupt_target)
        denom = (c_diff - z_diff).clamp_min(1e-6)
        rest = (p_diff - z_diff) / denom

        clean_pred = clean_out["logits"][:, -1, :].argmax(dim=-1)
        corrupt_pred = corrupt_out["logits"][:, -1, :].argmax(dim=-1)
        patched_pred = patched_out["logits"][:, -1, :].argmax(dim=-1)

        clean_diffs.extend(c_diff.detach().cpu().tolist())
        corrupt_diffs.extend(z_diff.detach().cpu().tolist())
        patched_diffs.extend(p_diff.detach().cpu().tolist())
        restorations.extend(rest.clamp(-5, 5).detach().cpu().tolist())
        clean_accs.append(float((clean_pred == clean_target).float().mean().item()))
        corrupt_accs.append(float((corrupt_pred == corrupt_target).float().mean().item()))
        patched_clean_accs.append(float((patched_pred == clean_target).float().mean().item()))

    out = {
        "clean_logit_diff_mean": summarize_vals(clean_diffs)["mean"],
        "corrupt_logit_diff_mean": summarize_vals(corrupt_diffs)["mean"],
        "patched_logit_diff_mean": summarize_vals(patched_diffs)["mean"],
        "restoration_mean": summarize_vals(restorations)["mean"],
        "restoration_stderr": summarize_vals(restorations)["stderr"],
        "clean_acc_mean": sum(clean_accs) / len(clean_accs),
        "corrupt_acc_mean": sum(corrupt_accs) / len(corrupt_accs),
        "patched_clean_answer_acc_mean": sum(patched_clean_accs) / len(patched_clean_accs),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Clean/corrupt activation patching for HOP_2 compositional behavior.")
    p.add_argument("run_dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None)
    p.add_argument("--head-patching", action="store_true", help="Also patch each individual head output at query position")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    model, data_cfg, model_cfg, cfg = load_model(run_dir, device)
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    base = eval_patch_site(model, data_cfg, model_cfg, rng, args.batch_size, args.num_batches, patch_kind="none")
    rows.append({"site": "corrupt_baseline", "patch_kind": "none", "layer": None, "head": None, **base})

    for layer in range(model_cfg.n_layers):
        rng_l = np.random.default_rng(args.seed + 1000 + layer)
        res = eval_patch_site(model, data_cfg, model_cfg, rng_l, args.batch_size, args.num_batches, patch_kind="residual", layer=layer)
        rows.append({"site": f"residual_L{layer}_query", "patch_kind": "residual", "layer": layer, "head": None, **res})

    if args.head_patching:
        for layer in range(model_cfg.n_layers):
            for head in range(model_cfg.n_heads):
                rng_h = np.random.default_rng(args.seed + 10000 + layer * 100 + head)
                res = eval_patch_site(model, data_cfg, model_cfg, rng_h, args.batch_size, args.num_batches, patch_kind="head", layer=layer, head=head)
                rows.append({"site": f"head_L{layer}H{head}_query", "patch_kind": "head", "layer": layer, "head": head, **res})

    out_path = Path(args.out) if args.out else run_dir / "hop2_activation_patching.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({"run_dir": str(run_dir), "args": vars(args), "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")
    # Print most restored residual sites for quick inspection.
    best = sorted([r for r in rows if r["patch_kind"] != "none"], key=lambda r: r.get("restoration_mean", -999), reverse=True)[:10]
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
