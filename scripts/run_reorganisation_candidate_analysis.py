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
import torch.nn.functional as F

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
    return run_dir.parent.name if run_dir.parent.name else run_dir.name


def load_seed(run_dir: Path, cfg: Dict[str, Any]) -> int | None:
    seed = cfg.get('train', {}).get('seed')
    if seed is not None:
        return int(seed)
    m = re.search(r'seed(\d+)', str(run_dir))
    return int(m.group(1)) if m else None


def checkpoint_step(path: Path, cfg: Dict[str, Any]) -> int:
    if path.name == 'checkpoint_pre_intro.pt':
        return int(cfg.get('train', {}).get('intro_step') or 0)
    m = re.search(r'checkpoint_pre_step_(\d+)\.pt', path.name)
    if m:
        return int(m.group(1))
    if path.name == 'model_final.pt':
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
    found: Dict[str, Path] = {}
    for p in run_dir.glob('checkpoint_pre_step_*.pt'):
        found[p.name] = p
    pi = run_dir / 'checkpoint_pre_intro.pt'
    if pi.exists():
        found[pi.name] = pi
    if include_final and (run_dir / 'model_final.pt').exists():
        found['model_final.pt'] = run_dir / 'model_final.pt'
    return list(found.values())


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


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


@torch.no_grad()
def hop2_lens_and_candidate_metrics(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    v_content = gen.cfg.v_content
    n_layers = model.cfg.n_layers
    accum: Dict[str, List[float]] = {}

    def add(key: str, val: torch.Tensor | float) -> None:
        if isinstance(val, torch.Tensor):
            x = float(val.detach().float().mean().item())
        else:
            x = float(val)
        accum.setdefault(key, []).append(x)

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
        bsz = batch.input_ids.shape[0]
        b_idx = torch.arange(bsz, device=device)
        qpos = batch.query_pos
        logits_full = out['logits'][b_idx, qpos, :]
        probs_full = F.softmax(logits_full, dim=-1)
        logits_content = logits_full[:, :v_content]
        probs_content = probs_full[:, :v_content]
        pred_full = logits_full.argmax(dim=-1)
        pred_content = logits_content.argmax(dim=-1)
        target = batch.target
        inter = batch.intermediate_token

        # Candidate masks over content vocabulary.
        context_mask = torch.zeros(bsz, v_content, dtype=torch.bool, device=device)
        value_mask = torch.zeros(bsz, v_content, dtype=torch.bool, device=device)
        content_ids = batch.input_ids.masked_fill(batch.input_ids >= v_content, 0)
        content_pos_mask = batch.input_ids < v_content
        context_mask.scatter_(1, content_ids, content_pos_mask)
        binding_len = (gen.cfg.chain_length - 1) * 3
        value_positions = torch.arange(1, binding_len, 3, device=device)
        value_ids = batch.input_ids[:, value_positions].clamp_max(v_content - 1)
        value_mask.scatter_(1, value_ids, True)

        add('final_target_acc_full_vocab', pred_full == target)
        add('final_target_acc_content_argmax', pred_content == target)
        add('final_intermediate_acc_content_argmax', pred_content == inter)
        add('top1_is_content_token', pred_full < v_content)
        add('top1_is_in_context_content', context_mask.gather(1, pred_content[:, None]).squeeze(1))
        add('top1_is_binding_value', value_mask.gather(1, pred_content[:, None]).squeeze(1))
        add('prob_content_vocab_mass', probs_content.sum(dim=-1))
        add('prob_in_context_content_mass', (probs_content * context_mask.float()).sum(dim=-1))
        add('prob_binding_value_mass', (probs_content * value_mask.float()).sum(dim=-1))
        add('prob_target', probs_full[b_idx, target])
        add('prob_intermediate', probs_full[b_idx, inter])
        # Among in-context candidates, did the model prefer the true answer?
        masked_context_logits = logits_content.masked_fill(~context_mask, -1e9)
        context_argmax = masked_context_logits.argmax(dim=-1)
        add('in_context_argmax_is_target', context_argmax == target)
        masked_value_logits = logits_content.masked_fill(~value_mask, -1e9)
        value_argmax = masked_value_logits.argmax(dim=-1)
        add('binding_value_argmax_is_target', value_argmax == target)
        cand_mass = (probs_content * context_mask.float()).sum(dim=-1).clamp_min(1e-12)
        cand_probs = (probs_content * context_mask.float()) / cand_mass[:, None]
        add('entropy_in_context_content', entropy_from_probs(cand_probs))
        val_mass = (probs_content * value_mask.float()).sum(dim=-1).clamp_min(1e-12)
        val_probs = (probs_content * value_mask.float()) / val_mass[:, None]
        add('entropy_binding_values', entropy_from_probs(val_probs))

        for layer, res in enumerate(out['residuals']):
            h = res[b_idx, qpos, :]
            lens_logits = model.lm_head(model.ln_f(h))[:, :v_content]
            lens_probs = F.softmax(lens_logits, dim=-1)
            lens_pred = lens_logits.argmax(dim=-1)
            add(f'lens_L{layer}_B_acc', lens_pred == inter)
            add(f'lens_L{layer}_C_acc', lens_pred == target)
            add(f'lens_L{layer}_B_prob', lens_probs[b_idx, inter])
            add(f'lens_L{layer}_C_prob', lens_probs[b_idx, target])
            add(f'lens_L{layer}_C_minus_B_logit', lens_logits[b_idx, target] - lens_logits[b_idx, inter])
            # Candidate format at each layer lens.
            add(f'lens_L{layer}_top1_is_in_context_content', context_mask.gather(1, lens_pred[:, None]).squeeze(1))
            add(f'lens_L{layer}_top1_is_binding_value', value_mask.gather(1, lens_pred[:, None]).squeeze(1))

    return {k: float(sum(v) / max(1, len(v))) for k, v in accum.items()}


def gini(vals: torch.Tensor) -> float:
    x = vals.detach().float().flatten().clamp_min(0).cpu()
    s = float(x.sum().item())
    if x.numel() == 0 or s <= 0:
        return 0.0
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32)
    return float(((2 * idx - n - 1) * x_sorted).sum().item() / (n * s))


def eff_heads(vals: torch.Tensor) -> float:
    x = vals.detach().float().flatten().clamp_min(0).cpu()
    s = float(x.sum().item())
    if s <= 0:
        return 0.0
    p = (x / s).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()).item())


def eval_checkpoint(
    model: TinyTransformer,
    data_cfg: DataConfig,
    optim_cfg: OptimConfig,
    sched_cfg: ScheduleConfig,
    cfg: Dict[str, Any],
    ckpt: Path,
    device: torch.device,
    batch_size: int,
    eval_batches: int,
    score_batches: int,
    lens_batches: int,
    gen_seed: int,
) -> Dict[str, Any]:
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    gen = ChainBatchGenerator(data_cfg, seed=gen_seed)
    step = checkpoint_step(ckpt, cfg)
    row: Dict[str, Any] = {
        'checkpoint': ckpt.name,
        'checkpoint_kind': checkpoint_kind(ckpt),
        'step': step,
        'lr_at_step': lr_at_step(max(0, step), optim_cfg, sched_cfg),
    }
    by_hop = evaluate_by_hop(model, gen, batch_size=batch_size, num_batches=eval_batches, device=device, k_max=data_cfg.k_max)
    row.update(by_hop)
    if data_cfg.k_max >= 2:
        floor2 = content_shuffled_floor(model, gen, batch_size=batch_size, num_batches=max(1, eval_batches // 2), device=device, force_hop=2)
        row['floor_hop2_acc'] = floor2
        row['hop2_excess'] = float(row.get('hop2_acc', 0.0) - floor2)
        scores = two_hop_attention_scores(model, gen, batch_size=batch_size, num_batches=score_batches, device=device)
        for name, tensor in scores.items():
            flat = tensor.flatten()
            row[f'{name}_max'] = float(flat.max().item())
            row[f'{name}_mean'] = float(flat.mean().item())
            row[f'{name}_top2_sum'] = float(torch.topk(flat, k=min(2, flat.numel())).values.sum().item())
            row[f'{name}_top4_sum'] = float(torch.topk(flat, k=min(4, flat.numel())).values.sum().item())
            row[f'{name}_gini'] = gini(tensor)
            row[f'{name}_eff_heads'] = eff_heads(tensor)
        if lens_batches > 0:
            row.update(hop2_lens_and_candidate_metrics(model, gen, batch_size=batch_size, num_batches=lens_batches, device=device))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description='v0.18: track HOP_2 reorganisation, two-hop scores, and candidate-set/format metrics across checkpoints.')
    p.add_argument('run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--score-batches', type=int, default=4)
    p.add_argument('--lens-batches', type=int, default=4)
    p.add_argument('--gen-seed', type=int, default=202618)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    model, data_cfg, _model_cfg, optim_cfg, sched_cfg, cfg = load_model_cfg(run_dir, device)
    ckpts = sorted(find_checkpoints(run_dir, include_final=True), key=lambda pth: (checkpoint_step(pth, cfg), pth.name))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {run_dir}')
    rows: List[Dict[str, Any]] = []
    arm = derive_arm(run_dir)
    seed = load_seed(run_dir, cfg)
    for ckpt in ckpts:
        row = eval_checkpoint(model, data_cfg, optim_cfg, sched_cfg, cfg, ckpt, device, args.batch_size, args.eval_batches, args.score_batches, args.lens_batches, args.gen_seed)
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
        print(f"{arm} seed={seed} step={row['step']} hop2={row.get('hop2_acc', float('nan')):.3f} B_L3={row.get('lens_L3_B_acc', float('nan')):.3f} C_L3={row.get('lens_L3_C_acc', float('nan')):.3f} cand_mass={row.get('prob_in_context_content_mass', float('nan')):.3f}")
    out = Path(args.out) if args.out else run_dir / 'reorganisation_candidate_tracking.csv'
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
