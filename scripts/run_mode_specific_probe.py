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
    force_hop: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Collect query-position residuals and labels for HOP_1 or HOP_2 prompts.

    Returns:
        features: [N, L, D]
        labels: dict of [N] tensors.
          HOP_1: hop1_target
          HOP_2: hop2_intermediate, hop2_answer
    """
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
        layer_feats = [res[b_idx, qpos, :].detach() for res in out['residuals']]
        feats.append(torch.stack(layer_feats, dim=1).cpu())
        if force_hop == 1:
            labels.setdefault('hop1_target', []).append(batch.target.detach().cpu())
        elif force_hop == 2:
            labels.setdefault('hop2_intermediate', []).append(batch.intermediate_token.detach().cpu())
            labels.setdefault('hop2_answer', []).append(batch.target.detach().cpu())
        else:
            raise ValueError('force_hop must be 1 or 2')
    return torch.cat(feats, dim=0), {k: torch.cat(v, dim=0) for k, v in labels.items()}


@torch.no_grad()
def logit_lens_stats(
    model: TinyTransformer,
    features: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    v_content: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    features = features.to(device)
    labels_dev = {k: v.to(device) for k, v in labels.items()}
    N, L, _D = features.shape
    b_idx = torch.arange(N, device=device)
    for layer in range(L):
        h = features[:, layer, :]
        logits = model.lm_head(model.ln_f(h))[:, :v_content]
        pred = logits.argmax(dim=-1)
        sorted_idx = logits.argsort(dim=-1, descending=True)
        row: Dict[str, float] = {'layer': float(layer)}
        for name, y in labels_dev.items():
            row[f'{name}_lens_acc'] = (pred == y).float().mean().item()
            nz = (sorted_idx == y[:, None]).nonzero()
            if nz.numel() > 0:
                row[f'{name}_lens_rank_mean'] = nz[:, 1].float().mean().item() + 1.0
            else:
                row[f'{name}_lens_rank_mean'] = float('nan')
            row[f'{name}_lens_logit_mean'] = logits[b_idx, y].mean().item()
        rows.append(row)
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
    mu = x_train.mean(dim=0, keepdim=True)
    sig = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    x_train = (x_train - mu) / sig
    x_eval = (x_eval - mu) / sig
    probe = LinearProbe(x_train.shape[-1], n_classes).to(device)
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
            'probe_train_acc': (train_logits.argmax(dim=-1) == y_train).float().mean().item(),
            'probe_eval_acc': (eval_logits.argmax(dim=-1) == y_eval).float().mean().item(),
            'probe_eval_loss': F.cross_entropy(eval_logits, y_eval).item(),
        }


def first_layer(rows: List[Dict[str, Any]], key: str, threshold: float) -> int | None:
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= threshold:
            return int(r['layer'])
    return None


def probe_mode(
    mode: str,
    model: TinyTransformer,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    device: torch.device,
    args: argparse.Namespace,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    force_hop = 1 if mode == 'hop1_prompt' else 2
    gen_train = ChainBatchGenerator(data_cfg, seed=args.seed + seed_offset)
    gen_eval = ChainBatchGenerator(data_cfg, seed=args.seed + 100000 + seed_offset)
    x_train, y_train = collect_features(model, gen_train, args.batch_size, args.train_batches, force_hop, device)
    x_eval, y_eval = collect_features(model, gen_eval, args.batch_size, args.eval_batches, force_hop, device)
    lens_rows = logit_lens_stats(model, x_eval, y_eval, data_cfg.v_content, device)
    rows: List[Dict[str, Any]] = []
    for r in lens_rows:
        row: Dict[str, Any] = {
            'mode': mode,
            'layer': int(r['layer']),
            'n_train': int(x_train.shape[0]),
            'n_eval': int(x_eval.shape[0]),
            **{k: v for k, v in r.items() if k != 'layer'},
        }
        rows.append(row)
    if not args.skip_linear_probe:
        for layer in range(model_cfg.n_layers):
            for label_name, ytr in y_train.items():
                yev = y_eval[label_name]
                stats = train_probe(
                    x_train[:, layer, :], ytr, x_eval[:, layer, :], yev,
                    n_classes=data_cfg.v_content,
                    steps=args.probe_steps,
                    lr=args.probe_lr,
                    weight_decay=args.probe_weight_decay,
                    device=device,
                )
                for k, v in stats.items():
                    rows[layer][f'{label_name}_{k}'] = v
    # Add per-mode first-layer summaries.
    label_names = sorted(y_train.keys())
    for label_name in label_names:
        for th_name, th in [('0p5', 0.5), ('0p8', 0.8)]:
            fl = first_layer(rows, f'{label_name}_probe_eval_acc', th)
            for row in rows:
                row[f'first_layer_{label_name}_probe_ge_{th_name}'] = fl
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='Mode-specific probe: compare HOP_1 prompt lookup representation with HOP_2 prompt intermediate/answer representations.')
    p.add_argument('run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--train-batches', type=int, default=16)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--probe-steps', type=int, default=300)
    p.add_argument('--probe-lr', type=float, default=1e-2)
    p.add_argument('--probe-weight-decay', type=float, default=1e-4)
    p.add_argument('--skip-linear-probe', action='store_true')
    p.add_argument('--seed', type=int, default=202615)
    p.add_argument('--probe-group', default='unknown')
    p.add_argument('--modes', default='hop1_prompt,hop2_prompt', help='comma separated subset: hop1_prompt,hop2_prompt')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    model, data_cfg, model_cfg, cfg = load_model(run_dir, device)
    if data_cfg.k_max < 2:
        raise ValueError('mode-specific probe requires k_max >= 2')
    seed = load_seed(run_dir, cfg)
    arm = derive_arm(run_dir)
    seed_offset = int(seed or 0) * 17

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    all_rows: List[Dict[str, Any]] = []
    for mode in modes:
        if mode not in {'hop1_prompt', 'hop2_prompt'}:
            raise ValueError(f'unknown mode {mode}')
        rows = probe_mode(mode, model, data_cfg, model_cfg, device, args, seed_offset + (0 if mode == 'hop1_prompt' else 999))
        for row in rows:
            row.update({
                'run_dir': str(run_dir),
                'arm': arm,
                'seed': seed,
                'probe_group': args.probe_group,
            })
        all_rows.extend(rows)

    # Cross-mode convenience summaries.
    def max_metric(mode: str, metric: str) -> float | None:
        vals = [r.get(metric) for r in all_rows if r.get('mode') == mode]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        return max(vals) if vals else None
    hop1_target = max_metric('hop1_prompt', 'hop1_target_probe_eval_acc')
    hop2_inter = max_metric('hop2_prompt', 'hop2_intermediate_probe_eval_acc')
    hop2_ans = max_metric('hop2_prompt', 'hop2_answer_probe_eval_acc')
    for row in all_rows:
        row['max_hop1_prompt_target_probe_acc'] = hop1_target
        row['max_hop2_prompt_intermediate_probe_acc'] = hop2_inter
        row['max_hop2_prompt_answer_probe_acc'] = hop2_ans
        if hop1_target is not None and hop2_inter is not None:
            row['hop1_target_minus_hop2_intermediate_probe_acc'] = hop1_target - hop2_inter
        if hop2_inter is not None and hop2_ans is not None:
            row['hop2_answer_minus_intermediate_probe_acc'] = hop2_ans - hop2_inter

    out_path = Path(args.out) if args.out else run_dir / 'mode_specific_probe.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
