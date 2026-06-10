#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

import torch

from cp_toy.config import DataConfig, ModelConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import compute_global_head_means, evaluate_by_hop, key_slot_lookup_scores, top_heads_from_scores
from cp_toy.model import TinyTransformer
from cp_toy.train import resolve_device


def main():
    p = argparse.ArgumentParser(description="Mean-ablation evaluation for identified key-slot heads")
    p.add_argument("run_dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--top-k", type=int, default=1)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg["data"])
    model_cfg = ModelConfig(**cfg["model"])
    device = resolve_device(args.device)
    model = TinyTransformer(model_cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model_final.pt", map_location=device))
    gen = ChainBatchGenerator(data_cfg, seed=999)

    scores = key_slot_lookup_scores(model, gen, args.batch_size, args.eval_batches, device)
    selected = top_heads_from_scores(scores, k=args.top_k)
    means = compute_global_head_means(model, gen, args.batch_size, max(2, args.eval_batches // 2), device)

    base = evaluate_by_hop(model, gen, args.batch_size, args.eval_batches, device, data_cfg.k_max)
    ablated = evaluate_by_hop(
        model,
        gen,
        args.batch_size,
        args.eval_batches,
        device,
        data_cfg.k_max,
        ablate_heads=selected,
        ablation_means=means,
    )
    out = {
        "selected_heads": selected,
        "keyslot_scores": scores.tolist(),
        "base": base,
        "ablated": ablated,
    }
    with open(run_dir / f"ablation_top{args.top_k}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
