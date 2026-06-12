#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path as _Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import torch

from cp_toy.config import DataConfig, ModelConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import compute_global_head_means, evaluate_by_hop, key_slot_lookup_scores, top_heads_from_scores
from cp_toy.model import HeadSelection, TinyTransformer
from cp_toy.train import resolve_device


def flatten_selection(sel: HeadSelection) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for layer, heads in sel.items():
        for h in heads:
            pairs.append((int(layer), int(h)))
    return sorted(pairs)


def selection_from_pairs(pairs: List[Tuple[int, int]]) -> HeadSelection:
    out: HeadSelection = {}
    for layer, head in pairs:
        out.setdefault(int(layer), []).append(int(head))
    for layer in out:
        out[layer] = sorted(out[layer])
    return out


def all_heads(n_layers: int, n_heads: int) -> List[Tuple[int, int]]:
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]


def random_selection(
    rng: random.Random,
    n_layers: int,
    n_heads: int,
    k: int,
    exclude: List[Tuple[int, int]] | None = None,
) -> HeadSelection:
    exclude_set = set(exclude or [])
    pool = [p for p in all_heads(n_layers, n_heads) if p not in exclude_set]
    if len(pool) < k:
        pool = all_heads(n_layers, n_heads)
    return selection_from_pairs(rng.sample(pool, k=min(k, len(pool))))


def eval_with_selection(model, gen, data_cfg, args, device, means, selection: HeadSelection) -> Dict[str, float]:
    return evaluate_by_hop(
        model,
        gen,
        args.batch_size,
        args.eval_batches,
        device,
        data_cfg.k_max,
        ablate_heads=selection,
        ablation_means=means,
    )


def acc_drop(base: Dict[str, float], ablated: Dict[str, float], hop: int) -> float | None:
    b = base.get(f'hop{hop}_acc')
    a = ablated.get(f'hop{hop}_acc')
    if b is None or a is None:
        return None
    return float(b) - float(a)


def summarize_draws(base: Dict[str, float], draws: List[Dict[str, object]], k_max: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for hop in range(1, k_max + 1):
        accs = [float(d['ablated'][f'hop{hop}_acc']) for d in draws if f'hop{hop}_acc' in d.get('ablated', {})]
        losses = [float(d['ablated'][f'hop{hop}_loss']) for d in draws if f'hop{hop}_loss' in d.get('ablated', {})]
        drops = [float(base[f'hop{hop}_acc']) - a for a in accs] if f'hop{hop}_acc' in base else []
        if accs:
            out[f'random_hop{hop}_acc_mean'] = sum(accs) / len(accs)
            out[f'random_hop{hop}_acc_min'] = min(accs)
            out[f'random_hop{hop}_acc_max'] = max(accs)
        if losses:
            out[f'random_hop{hop}_loss_mean'] = sum(losses) / len(losses)
        if drops:
            out[f'random_hop{hop}_drop_mean'] = sum(drops) / len(drops)
            out[f'random_hop{hop}_drop_max'] = max(drops)
            out[f'random_hop{hop}_drop_min'] = min(drops)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='Matched random-head control for key-slot mean ablation.')
    p.add_argument('run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batches', type=int, default=16)
    p.add_argument('--top-k', type=int, default=4)
    p.add_argument('--num-random', type=int, default=20)
    p.add_argument('--random-seed', type=int, default=12345)
    p.add_argument('--include-selected-in-random-pool', action='store_true', help='By default random draws exclude selected key-slot heads.')
    p.add_argument('--calib-batches', type=int, default=None, help='Batches for head means/key-slot scores; defaults to max(2, eval_batches//2).')
    args = p.parse_args()

    run_dir = _Path(args.run_dir)
    with open(run_dir / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg['data'])
    model_cfg = ModelConfig(**cfg['model'])
    device = resolve_device(args.device)
    model = TinyTransformer(model_cfg).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_final.pt', map_location=device))
    gen = ChainBatchGenerator(data_cfg, seed=999)

    calib_batches = args.calib_batches if args.calib_batches is not None else max(2, args.eval_batches // 2)
    scores = key_slot_lookup_scores(model, gen, args.batch_size, calib_batches, device)
    selected = top_heads_from_scores(scores, k=args.top_k)
    selected_pairs = flatten_selection(selected)
    means = compute_global_head_means(model, gen, args.batch_size, calib_batches, device)

    base = evaluate_by_hop(model, gen, args.batch_size, args.eval_batches, device, data_cfg.k_max)
    keyslot_ablated = eval_with_selection(model, gen, data_cfg, args, device, means, selected)

    rng = random.Random(args.random_seed)
    random_draws: List[Dict[str, object]] = []
    exclude = [] if args.include_selected_in_random_pool else selected_pairs
    for draw_idx in range(args.num_random):
        sel = random_selection(rng, model_cfg.n_layers, model_cfg.n_heads, args.top_k, exclude=exclude)
        ab = eval_with_selection(model, gen, data_cfg, args, device, means, sel)
        random_draws.append({
            'draw_idx': draw_idx,
            'selected_heads': sel,
            'ablated': ab,
        })

    summary = summarize_draws(base, random_draws, data_cfg.k_max)
    for hop in range(1, data_cfg.k_max + 1):
        kd = acc_drop(base, keyslot_ablated, hop)
        if kd is not None:
            summary[f'keyslot_hop{hop}_drop'] = kd
            rd = summary.get(f'random_hop{hop}_drop_mean')
            if rd is not None:
                summary[f'keyslot_minus_random_hop{hop}_drop'] = kd - float(rd)
            summary[f'keyslot_hop{hop}_ablated_acc'] = float(keyslot_ablated.get(f'hop{hop}_acc'))
            summary[f'base_hop{hop}_acc'] = float(base.get(f'hop{hop}_acc'))

    out = {
        'run_dir': str(run_dir),
        'top_k': args.top_k,
        'num_random': args.num_random,
        'random_seed': args.random_seed,
        'random_excludes_selected': not args.include_selected_in_random_pool,
        'selected_heads': selected,
        'selected_pairs': selected_pairs,
        'keyslot_scores': scores.tolist(),
        'base': base,
        'keyslot_ablated': keyslot_ablated,
        'random_draws': random_draws,
        'summary': summary,
    }
    out_path = run_dir / f'matched_random_ablation_top{args.top_k}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps({k: out[k] for k in ['run_dir', 'top_k', 'num_random', 'random_excludes_selected', 'selected_heads', 'base', 'keyslot_ablated', 'summary']}, indent=2, sort_keys=True))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
