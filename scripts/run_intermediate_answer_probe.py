#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cp_toy.config import DataConfig, ModelConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.model import TinyTransformer
from cp_toy.train import resolve_device

ARM_NAMES = {
    's1_late_original', 's1_plateau_late', 's1_longcos_late', 's2_constant_late',
    'rewarm_late', 'rewarm_reset_late', 'fresh_hop1_s1', 'fresh_hop1_s2',
}


def load_model(run_dir: Path, device: torch.device) -> Tuple[TinyTransformer, DataConfig, ModelConfig, Dict[str, Any]]:
    with open(run_dir / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg['data'])
    model_cfg = ModelConfig(**cfg['model'])
    model = TinyTransformer(model_cfg).to(device)
    state = torch.load(run_dir / 'model_final.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, data_cfg, model_cfg, cfg


def derive_arm(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ARM_NAMES:
            return part
    return run_dir.parent.name if run_dir.parent.name in ARM_NAMES else run_dir.name


def load_seed(run_dir: Path, cfg: Dict[str, Any]) -> int | None:
    seed = cfg.get('train', {}).get('seed')
    if seed is not None:
        return int(seed)
    m = re.search(r'seed(\d+)', run_dir.name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def collect_features(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect query-position residuals for forced HOP_2 examples.

    Returns:
        features: [N, L, D]
        intermediate: [N]
        answer: [N]
    """
    feats: List[torch.Tensor] = []
    inters: List[torch.Tensor] = []
    answers: List[torch.Tensor] = []
    for _ in range(num_batches):
        batch = gen.batch(
            batch_size=batch_size,
            p_dynamic=1.0,
            p_multi=0.0,
            force_dynamic=True,
            force_hop=2,
            query_marker='A',
            token_pool='all',
            device=device,
        )
        out = model(batch.input_ids, return_residuals=True)
        b_idx = torch.arange(batch.input_ids.shape[0], device=device)
        # query_pos is constant by construction but keep batched indexing.
        qpos = batch.query_pos
        layer_feats = [res[b_idx, qpos, :].detach() for res in out['residuals']]
        feats.append(torch.stack(layer_feats, dim=1).cpu())  # B,L,D
        inters.append(batch.intermediate_token.detach().cpu())
        answers.append(batch.target.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(inters, dim=0), torch.cat(answers, dim=0)


@torch.no_grad()
def logit_lens_stats(model: TinyTransformer, features: torch.Tensor, inter: torch.Tensor, ans: torch.Tensor, v_content: int, device: torch.device) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    features = features.to(device)
    inter = inter.to(device)
    ans = ans.to(device)
    N, L, _D = features.shape
    b_idx = torch.arange(N, device=device)
    for layer in range(L):
        h = features[:, layer, :]
        logits = model.lm_head(model.ln_f(h))[:, :v_content]
        pred = logits.argmax(dim=-1)
        inter_acc = (pred == inter).float().mean().item()
        ans_acc = (pred == ans).float().mean().item()
        # Ranks: 1 is best.
        sorted_idx = logits.argsort(dim=-1, descending=True)
        inter_rank = (sorted_idx == inter[:, None]).nonzero()[:, 1].float().mean().item() + 1.0
        ans_rank = (sorted_idx == ans[:, None]).nonzero()[:, 1].float().mean().item() + 1.0
        rows.append({
            'layer': float(layer),
            'lens_intermediate_acc': inter_acc,
            'lens_answer_acc': ans_acc,
            'lens_intermediate_rank_mean': inter_rank,
            'lens_answer_rank_mean': ans_rank,
            'lens_intermediate_logit_mean': logits[b_idx, inter].mean().item(),
            'lens_answer_logit_mean': logits[b_idx, ans].mean().item(),
            'lens_answer_minus_intermediate_logit_mean': (logits[b_idx, ans] - logits[b_idx, inter]).mean().item(),
        })
    return rows


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    n_classes: int,
    steps: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, float]:
    x_train = x_train.to(device).float()
    y_train = y_train.to(device).long()
    x_eval = x_eval.to(device).float()
    y_eval = y_eval.to(device).long()
    # Standardise features using train stats. This makes probe training less sensitive
    # to residual scale differences across layers/runs.
    mu = x_train.mean(dim=0, keepdim=True)
    sig = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    x_train = (x_train - mu) / sig
    x_eval = (x_eval - mu) / sig
    probe = LinearProbe(x_train.shape[-1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    # Full-batch optimisation is stable and tiny here.
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_logits = probe(x_train)
        eval_logits = probe(x_eval)
        train_acc = (train_logits.argmax(dim=-1) == y_train).float().mean().item()
        eval_acc = (eval_logits.argmax(dim=-1) == y_eval).float().mean().item()
        eval_loss = F.cross_entropy(eval_logits, y_eval).item()
    return {'probe_train_acc': train_acc, 'probe_eval_acc': eval_acc, 'probe_eval_loss': eval_loss}


def first_layer_at_or_above(rows: List[Dict[str, Any]], key: str, threshold: float) -> int | None:
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and float(v) >= threshold:
            return int(r['layer'])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description='E2: probe intermediate B and answer C decodability at the query position across layers.')
    p.add_argument('run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--train-batches', type=int, default=16)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--probe-steps', type=int, default=300)
    p.add_argument('--probe-lr', type=float, default=1e-2)
    p.add_argument('--probe-weight-decay', type=float, default=1e-4)
    p.add_argument('--skip-linear-probe', action='store_true')
    p.add_argument('--seed', type=int, default=202614)
    p.add_argument('--probe-group', default='unknown')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    model, data_cfg, model_cfg, cfg = load_model(run_dir, device)
    if data_cfg.k_max < 2:
        raise ValueError('intermediate/answer probe requires k_max >= 2')
    seed = load_seed(run_dir, cfg)
    arm = derive_arm(run_dir)
    gen_train = ChainBatchGenerator(data_cfg, seed=args.seed + int(seed or 0) * 17)
    gen_eval = ChainBatchGenerator(data_cfg, seed=args.seed + 100000 + int(seed or 0) * 17)

    x_train, yb_train, yc_train = collect_features(model, gen_train, args.batch_size, args.train_batches, device)
    x_eval, yb_eval, yc_eval = collect_features(model, gen_eval, args.batch_size, args.eval_batches, device)

    lens_rows = logit_lens_stats(model, x_eval, yb_eval, yc_eval, data_cfg.v_content, device)

    rows: List[Dict[str, Any]] = []
    for r in lens_rows:
        row: Dict[str, Any] = {
            'run_dir': str(run_dir),
            'arm': arm,
            'seed': seed,
            'probe_group': args.probe_group,
            'layer': int(r['layer']),
            'n_train': int(x_train.shape[0]),
            'n_eval': int(x_eval.shape[0]),
            **{k: v for k, v in r.items() if k != 'layer'},
        }
        rows.append(row)

    if not args.skip_linear_probe:
        for layer in range(model_cfg.n_layers):
            for label_name, ytr, yev in [
                ('intermediate', yb_train, yb_eval),
                ('answer', yc_train, yc_eval),
            ]:
                stats = train_probe(
                    x_train[:, layer, :], ytr,
                    x_eval[:, layer, :], yev,
                    n_classes=data_cfg.v_content,
                    steps=args.probe_steps,
                    lr=args.probe_lr,
                    weight_decay=args.probe_weight_decay,
                    device=device,
                )
                row = rows[layer]
                for k, v in stats.items():
                    row[f'{label_name}_{k}'] = v

    # Add simple within-run summary columns to every row for easy aggregation.
    inter_first_80 = first_layer_at_or_above(rows, 'intermediate_probe_eval_acc', 0.80)
    ans_first_80 = first_layer_at_or_above(rows, 'answer_probe_eval_acc', 0.80)
    inter_first_50 = first_layer_at_or_above(rows, 'intermediate_probe_eval_acc', 0.50)
    ans_first_50 = first_layer_at_or_above(rows, 'answer_probe_eval_acc', 0.50)
    for row in rows:
        row['first_layer_intermediate_probe_ge_0p5'] = inter_first_50
        row['first_layer_answer_probe_ge_0p5'] = ans_first_50
        row['first_layer_intermediate_probe_ge_0p8'] = inter_first_80
        row['first_layer_answer_probe_ge_0p8'] = ans_first_80
        if inter_first_80 is not None and ans_first_80 is not None:
            row['answer_minus_intermediate_first_0p8_layer'] = ans_first_80 - inter_first_80
        else:
            row['answer_minus_intermediate_first_0p8_layer'] = None

    out_path = Path(args.out) if args.out else run_dir / 'intermediate_answer_probe.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
