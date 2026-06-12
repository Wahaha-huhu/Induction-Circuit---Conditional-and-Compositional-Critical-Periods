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
    'rewarm_late', 'rewarm_reset_late', 'fresh_hop1_s1', 'fresh_hop1_s2'
}


def derive_arm(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ARM_NAMES:
            return part
    return run_dir.parent.name if run_dir.parent.name in ARM_NAMES else run_dir.name


def load_seed(run_dir: Path, cfg: Dict[str, Any]) -> int | None:
    seed = cfg.get('train', {}).get('seed')
    if seed is not None:
        return int(seed)
    m = re.search(r'seed(\d+)', str(run_dir))
    return int(m.group(1)) if m else None


def load_model(run_dir: Path, device: torch.device) -> Tuple[TinyTransformer, DataConfig, ModelConfig, Dict[str, Any]]:
    with open(run_dir / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg['data'])
    model_cfg = ModelConfig(**cfg['model'])
    model = TinyTransformer(model_cfg).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_final.pt', map_location=device))
    model.eval()
    return model, data_cfg, model_cfg, cfg


@torch.no_grad()
def collect_features(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    force_hop: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    feats: List[torch.Tensor] = []
    labels: Dict[str, List[torch.Tensor]] = {}
    for _ in range(num_batches):
        batch = gen.batch(
            batch_size=batch_size,
            p_dynamic=1.0,
            p_multi=0.0,
            force_dynamic=True,
            force_hop=force_hop,
            query_marker='A',
            token_pool='all',
            device=device,
        )
        out = model(batch.input_ids, return_residuals=True)
        b_idx = torch.arange(batch.input_ids.shape[0], device=device)
        qpos = batch.query_pos
        feats.append(torch.stack([res[b_idx, qpos, :].detach().cpu() for res in out['residuals']], dim=1))
        if force_hop == 1:
            labels.setdefault('hop1_target_B', []).append(batch.target.detach().cpu())
        elif force_hop == 2:
            labels.setdefault('hop2_intermediate_B', []).append(batch.intermediate_token.detach().cpu())
            labels.setdefault('hop2_answer_C', []).append(batch.target.detach().cpu())
    return torch.cat(feats, dim=0), {k: torch.cat(v, dim=0) for k, v in labels.items()}


class Probe(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)
    def forward(self, x):
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
    mu = x_train.mean(dim=0, keepdim=True)
    sig = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    x_train = (x_train - mu) / sig
    x_eval = (x_eval - mu) / sig
    probe = Probe(x_train.shape[-1], n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(x_train), y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_logits = probe(x_train)
        eval_logits = probe(x_eval)
        return {
            'train_acc': (train_logits.argmax(dim=-1) == y_train).float().mean().item(),
            'eval_acc': (eval_logits.argmax(dim=-1) == y_eval).float().mean().item(),
            'eval_loss': F.cross_entropy(eval_logits, y_eval).item(),
        }


@torch.no_grad()
def logit_lens(model: TinyTransformer, features: torch.Tensor, labels: Dict[str, torch.Tensor], v_content: int, device: torch.device) -> List[Dict[str, Any]]:
    features = features.to(device)
    labels_d = {k: v.to(device) for k, v in labels.items()}
    N, L, _ = features.shape
    b_idx = torch.arange(N, device=device)
    rows: List[Dict[str, Any]] = []
    for layer in range(L):
        logits = model.lm_head(model.ln_f(features[:, layer, :]))[:, :v_content]
        pred = logits.argmax(dim=-1)
        row: Dict[str, Any] = {'layer': layer}
        for name, y in labels_d.items():
            row[f'{name}_lens_acc'] = (pred == y).float().mean().item()
            row[f'{name}_lens_logit_mean'] = logits[b_idx, y].mean().item()
        rows.append(row)
    return rows


def run_for_model(run_dir: Path, args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    model, data_cfg, model_cfg, cfg = load_model(run_dir, device)
    arm = derive_arm(run_dir)
    seed = load_seed(run_dir, cfg)
    rows: List[Dict[str, Any]] = []
    for mode, force_hop, offset in [('hop1_prompt', 1, 11), ('hop2_prompt', 2, 22)]:
        gen_train = ChainBatchGenerator(data_cfg, seed=args.seed + offset)
        gen_eval = ChainBatchGenerator(data_cfg, seed=args.seed + 100000 + offset)
        x_train, y_train = collect_features(model, gen_train, args.batch_size, args.train_batches, force_hop, device)
        x_eval, y_eval = collect_features(model, gen_eval, args.batch_size, args.eval_batches, force_hop, device)
        lens_rows = logit_lens(model, x_eval, y_eval, data_cfg.v_content, device)
        lens_by_layer = {int(r['layer']): r for r in lens_rows}
        for layer in range(model_cfg.n_layers):
            for label_name, ytr in y_train.items():
                yev = y_eval[label_name]
                true_stats = train_probe(x_train[:, layer, :], ytr, x_eval[:, layer, :], yev, data_cfg.v_content, args.probe_steps, args.probe_lr, args.probe_weight_decay, device)
                # Shuffled-label control: train on randomised labels but evaluate on true held-out labels.
                gen = torch.Generator().manual_seed(args.seed + 4242 + layer + len(label_name))
                perm = torch.randperm(ytr.numel(), generator=gen)
                shuffled_stats = train_probe(x_train[:, layer, :], ytr[perm], x_eval[:, layer, :], yev, data_cfg.v_content, args.probe_steps, args.probe_lr, args.probe_weight_decay, device)
                # Frequency baseline: most common train label.
                bincount = torch.bincount(ytr, minlength=data_cfg.v_content)
                majority = int(bincount.argmax().item())
                majority_acc = (yev == majority).float().mean().item()
                row = {
                    'run_dir': str(run_dir),
                    'arm': arm,
                    'seed': seed,
                    'mode': mode,
                    'layer': layer,
                    'label_name': label_name,
                    'n_train': int(x_train.shape[0]),
                    'n_eval': int(x_eval.shape[0]),
                    'true_probe_train_acc': true_stats['train_acc'],
                    'true_probe_eval_acc': true_stats['eval_acc'],
                    'true_probe_eval_loss': true_stats['eval_loss'],
                    'shuffled_label_probe_eval_acc': shuffled_stats['eval_acc'],
                    'shuffled_label_probe_eval_loss': shuffled_stats['eval_loss'],
                    'probe_acc_above_shuffle': true_stats['eval_acc'] - shuffled_stats['eval_acc'],
                    'majority_baseline_acc': majority_acc,
                }
                for k, v in lens_by_layer[layer].items():
                    if k != 'layer':
                        row[k] = v
                rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='v0.18 mode-specific probe with shuffled-label controls.')
    p.add_argument('--run-dir', action='append', default=[])
    p.add_argument('--selected-runs', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--train-batches', type=int, default=16)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--probe-steps', type=int, default=300)
    p.add_argument('--probe-lr', type=float, default=1e-2)
    p.add_argument('--probe-weight-decay', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=202618)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    run_dirs = [Path(p) for p in args.run_dir]
    if args.selected_runs:
        run_dirs.extend(Path(line.strip()) for line in Path(args.selected_runs).read_text().splitlines() if line.strip())
    if not run_dirs:
        raise ValueError('No run dirs provided; use --run-dir or --selected-runs')
    device = resolve_device(args.device)
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        print(f'==> probe controls {rd}')
        rows.extend(run_for_model(rd, args, device))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
