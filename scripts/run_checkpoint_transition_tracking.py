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

from cp_toy.config import DataConfig, ModelConfig, OptimConfig, ScheduleConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import content_shuffled_floor, evaluate_by_hop, two_hop_attention_scores
from cp_toy.model import TinyTransformer
from cp_toy.schedules import lr_at_step
from cp_toy.train import resolve_device

ARM_NAMES = {
    's1_late_original', 's1_plateau_late', 's1_longcos_late', 's2_constant_late',
    'rewarm_late', 'rewarm_reset_late', 'late_gate_post_seed0', 'late_gate_post_seed1', 'late_gate_post_seed2'
}


def derive_arm(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in ARM_NAMES:
            return part
    # Common layout: .../<arm>/<condition_seedN>
    if run_dir.parent.name:
        return run_dir.parent.name
    return run_dir.name


def load_seed(run_dir: Path, cfg: Dict[str, Any]) -> int | None:
    seed = cfg.get('train', {}).get('seed')
    if seed is not None:
        return int(seed)
    m = re.search(r'seed(\d+)', str(run_dir))
    return int(m.group(1)) if m else None


def checkpoint_step(path: Path, cfg: Dict[str, Any]) -> int:
    name = path.name
    if name == 'checkpoint_pre_intro.pt':
        return int(cfg.get('train', {}).get('intro_step') or 0)
    m = re.search(r'checkpoint_pre_step_(\d+)\.pt', name)
    if m:
        return int(m.group(1))
    if name == 'model_final.pt':
        return int(cfg.get('train', {}).get('max_steps') or -1)
    return -1


def checkpoint_kind(path: Path) -> str:
    if path.name == 'checkpoint_pre_intro.pt':
        return 'pre_intro'
    if path.name.startswith('checkpoint_pre_step_'):
        return 'pre_step'
    if path.name == 'model_final.pt':
        return 'final'
    return 'unknown'


def find_checkpoints(run_dir: Path, include_final: bool = True) -> List[Path]:
    ckpts: Dict[str, Path] = {}
    for p in run_dir.glob('checkpoint_pre_step_*.pt'):
        ckpts[p.name] = p
    pi = run_dir / 'checkpoint_pre_intro.pt'
    if pi.exists():
        ckpts[pi.name] = pi
    if include_final and (run_dir / 'model_final.pt').exists():
        ckpts['model_final.pt'] = run_dir / 'model_final.pt'
    return list(ckpts.values())


def gini(vals: torch.Tensor) -> float:
    x = vals.detach().float().flatten().clamp_min(0).cpu()
    if x.numel() == 0:
        return float('nan')
    s = float(x.sum().item())
    if s <= 0:
        return 0.0
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float(((2 * idx - n - 1) * x_sorted).sum().item() / (n * s))


def entropy_eff_heads(vals: torch.Tensor) -> float:
    x = vals.detach().float().flatten().clamp_min(0).cpu()
    s = float(x.sum().item())
    if s <= 0:
        return 0.0
    p = (x / s).clamp_min(1e-12)
    ent = float(-(p * torch.log(p)).sum().item())
    return float(math.exp(ent))


@torch.no_grad()
def logit_lens_hop2(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    rows: Dict[str, List[float]] = {}
    v_content = gen.cfg.v_content
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
        qpos = batch.query_pos
        inter = batch.intermediate_token
        ans = batch.target
        for layer, res in enumerate(out['residuals']):
            h = res[b_idx, qpos, :]
            logits = model.lm_head(model.ln_f(h))[:, :v_content]
            pred = logits.argmax(dim=-1)
            for key, val in [
                (f'lens_L{layer}_B_acc', (pred == inter).float().mean().item()),
                (f'lens_L{layer}_C_acc', (pred == ans).float().mean().item()),
                (f'lens_L{layer}_C_minus_B_logit', (logits[b_idx, ans] - logits[b_idx, inter]).mean().item()),
            ]:
                rows.setdefault(key, []).append(float(val))
    return {k: float(sum(v) / max(1, len(v))) for k, v in rows.items()}


def load_model_cfg(run_dir: Path, device: torch.device):
    with (run_dir / 'config.json').open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg['data'])
    model_cfg = ModelConfig(**cfg['model'])
    optim_cfg = OptimConfig(**cfg['optim'])
    sched_cfg = ScheduleConfig(**cfg['schedule'])
    model = TinyTransformer(model_cfg).to(device)
    model.eval()
    return model, data_cfg, model_cfg, optim_cfg, sched_cfg, cfg


def eval_checkpoint(
    model: TinyTransformer,
    data_cfg: DataConfig,
    optim_cfg: OptimConfig,
    sched_cfg: ScheduleConfig,
    cfg: Dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
    eval_batches: int,
    score_batches: int,
    lens_batches: int,
    gen_seed: int,
) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    gen = ChainBatchGenerator(data_cfg, seed=gen_seed)
    step = checkpoint_step(ckpt_path, cfg)
    row: Dict[str, Any] = {
        'checkpoint': ckpt_path.name,
        'checkpoint_kind': checkpoint_kind(ckpt_path),
        'step': step,
        'lr_at_step': lr_at_step(max(0, step), optim_cfg, sched_cfg),
    }
    by_hop = evaluate_by_hop(
        model, gen, batch_size=batch_size, num_batches=eval_batches,
        device=device, k_max=data_cfg.k_max, query_marker='A', token_pool='all'
    )
    row.update(by_hop)
    if data_cfg.k_max >= 2:
        floor2 = content_shuffled_floor(
            model, gen, batch_size=batch_size, num_batches=max(1, eval_batches // 2),
            device=device, force_hop=2, token_pool='all'
        )
        row['floor_hop2_acc'] = floor2
        row['hop2_excess'] = float(row.get('hop2_acc', 0.0) - floor2)
        scores = two_hop_attention_scores(
            model, gen, batch_size=batch_size, num_batches=score_batches, device=device, token_pool='all'
        )
        for name, tensor in scores.items():
            flat = tensor.flatten()
            row[f'{name}_max'] = float(flat.max().item())
            row[f'{name}_mean'] = float(flat.mean().item())
            row[f'{name}_top2_sum'] = float(torch.topk(flat, k=min(2, flat.numel())).values.sum().item())
            row[f'{name}_top4_sum'] = float(torch.topk(flat, k=min(4, flat.numel())).values.sum().item())
            row[f'{name}_gini'] = gini(tensor)
            row[f'{name}_eff_heads'] = entropy_eff_heads(tensor)
            top_idx = int(flat.argmax().item())
            row[f'{name}_top_layer'] = top_idx // model.cfg.n_heads
            row[f'{name}_top_head'] = top_idx % model.cfg.n_heads
        if lens_batches > 0:
            row.update(logit_lens_hop2(model, gen, batch_size=batch_size, num_batches=lens_batches, device=device))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description='E4: evaluate HOP_2 accuracy, two-hop attention scores, and logit-lens representations across saved checkpoints.')
    p.add_argument('run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--score-batches', type=int, default=4)
    p.add_argument('--lens-batches', type=int, default=4)
    p.add_argument('--gen-seed', type=int, default=202617)
    p.add_argument('--include-final', action='store_true', default=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    model, data_cfg, model_cfg, optim_cfg, sched_cfg, cfg = load_model_cfg(run_dir, device)
    ckpts = find_checkpoints(run_dir, include_final=args.include_final)
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {run_dir}. Expected checkpoint_pre_step_*.pt and/or model_final.pt')
    ckpts = sorted(ckpts, key=lambda pth: (checkpoint_step(pth, cfg), pth.name))
    rows: List[Dict[str, Any]] = []
    arm = derive_arm(run_dir)
    seed = load_seed(run_dir, cfg)
    for ckpt in ckpts:
        row = eval_checkpoint(
            model, data_cfg, optim_cfg, sched_cfg, cfg, ckpt, device,
            batch_size=args.batch_size, eval_batches=args.eval_batches,
            score_batches=args.score_batches, lens_batches=args.lens_batches,
            gen_seed=args.gen_seed,
        )
        row.update({
            'run_dir': str(run_dir),
            'arm': arm,
            'seed': seed,
            'schedule_kind': cfg.get('schedule', {}).get('kind'),
            'intro_step': cfg.get('train', {}).get('intro_step'),
            'max_steps': cfg.get('train', {}).get('max_steps'),
            't_schedule': cfg.get('schedule', {}).get('t_schedule'),
        })
        rows.append(row)
        print(f"{arm} seed={seed} {ckpt.name}: step={row['step']} hop2={row.get('hop2_acc', float('nan')):.3f} excess={row.get('hop2_excess', float('nan')):.3f} second_key_max={row.get('second_key_max', float('nan')):.3f}")

    out = Path(args.out) if args.out else run_dir / 'transition_tracking.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
