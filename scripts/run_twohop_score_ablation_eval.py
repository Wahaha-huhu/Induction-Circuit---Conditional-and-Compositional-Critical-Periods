#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path as _Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import torch

from cp_toy.config import DataConfig, ModelConfig
from cp_toy.data import ChainBatchGenerator
from cp_toy.metrics import (
    compute_global_head_means,
    evaluate_by_hop,
    top_heads_from_scores,
    two_hop_attention_scores,
)
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


def random_selection(rng: random.Random, n_layers: int, n_heads: int, k: int, exclude: List[Tuple[int, int]]) -> HeadSelection:
    pool = [p for p in all_heads(n_layers, n_heads) if p not in set(exclude)]
    if len(pool) < k:
        pool = all_heads(n_layers, n_heads)
    return selection_from_pairs(rng.sample(pool, k=min(k, len(pool))))


def flat_head_selection(layer: int, head: int) -> HeadSelection:
    return {int(layer): [int(head)]}


def selection_str(sel: HeadSelection) -> str:
    return ','.join(f'L{l}H{h}' for l, h in flatten_selection(sel))


def eval_with_selection(model, gen, data_cfg, args, device, means, selection: HeadSelection) -> Dict[str, float]:
    return evaluate_by_hop(
        model, gen, args.batch_size, args.eval_batches, device, data_cfg.k_max,
        ablate_heads=selection, ablation_means=means,
    )


def acc_drop(base: Dict[str, float], ablated: Dict[str, float], hop: int) -> float | None:
    b = base.get(f'hop{hop}_acc')
    a = ablated.get(f'hop{hop}_acc')
    if b is None or a is None:
        return None
    return float(b) - float(a)


def summarize_random(base: Dict[str, float], draws: List[Dict[str, Any]], k_max: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for hop in range(1, k_max + 1):
        accs = []
        drops = []
        losses = []
        for d in draws:
            ab = d.get('ablated', {})
            if f'hop{hop}_acc' in ab:
                a = float(ab[f'hop{hop}_acc'])
                accs.append(a)
                if f'hop{hop}_acc' in base:
                    drops.append(float(base[f'hop{hop}_acc']) - a)
            if f'hop{hop}_loss' in ab:
                losses.append(float(ab[f'hop{hop}_loss']))
        if accs:
            out[f'random_hop{hop}_acc_mean'] = sum(accs) / len(accs)
            out[f'random_hop{hop}_acc_min'] = min(accs)
            out[f'random_hop{hop}_acc_max'] = max(accs)
        if drops:
            out[f'random_hop{hop}_drop_mean'] = sum(drops) / len(drops)
            out[f'random_hop{hop}_drop_min'] = min(drops)
            out[f'random_hop{hop}_drop_max'] = max(drops)
        if losses:
            out[f'random_hop{hop}_loss_mean'] = sum(losses) / len(losses)
    return out


def load_model(run_dir: _Path, device: torch.device) -> Tuple[TinyTransformer, DataConfig, ModelConfig, Dict[str, Any]]:
    with open(run_dir / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = DataConfig(**cfg['data'])
    model_cfg = ModelConfig(**cfg['model'])
    model = TinyTransformer(model_cfg).to(device)
    state = torch.load(run_dir / 'model_final.pt', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, data_cfg, model_cfg, cfg


def derive_label(run_dir: _Path) -> str:
    parts = run_dir.parts
    # Common layout: .../<arm>/<condition_seedX>
    if len(parts) >= 2:
        arm = parts[-2]
        seed = 'seed'
        m = re.search(r'seed(\d+)', run_dir.name)
        if m:
            seed = f'seed{m.group(1)}'
        return f'{arm}_{seed}'
    return run_dir.name


def main() -> None:
    p = argparse.ArgumentParser(description='E1: first-hop vs second-hop attention-score ablation with matched random controls.')
    p.add_argument('run_dir')
    p.add_argument('--out-dir', default=None, help='Directory for JSON output; default writes inside run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batches', type=int, default=16)
    p.add_argument('--calib-batches', type=int, default=None)
    p.add_argument('--top-k', type=int, default=4)
    p.add_argument('--num-random', type=int, default=20)
    p.add_argument('--random-seed', type=int, default=12345)
    p.add_argument('--include-causal-ranking', action='store_true', help='Also rank heads by single-head HOP_2 drops. More expensive.')
    p.add_argument('--include-selected-in-random-pool', action='store_true', help='By default random draws exclude all score-selected heads.')
    args = p.parse_args()

    run_dir = _Path(args.run_dir)
    out_dir = _Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, data_cfg, model_cfg, cfg = load_model(run_dir, device)
    if data_cfg.k_max < 2:
        raise ValueError('two-hop decomposition ablation requires k_max >= 2')
    seed = int(cfg.get('train', {}).get('seed', 0))
    gen = ChainBatchGenerator(data_cfg, seed=202613 + seed)
    calib_batches = args.calib_batches if args.calib_batches is not None else max(2, args.eval_batches // 2)

    scores = two_hop_attention_scores(model, gen, args.batch_size, calib_batches, device)
    selections: Dict[str, HeadSelection] = {
        name: top_heads_from_scores(score, k=args.top_k) for name, score in scores.items()
    }

    means = compute_global_head_means(model, gen, args.batch_size, calib_batches, device)
    base = evaluate_by_hop(model, gen, args.batch_size, args.eval_batches, device, data_cfg.k_max)

    ablated: Dict[str, Dict[str, float]] = {}
    summary: Dict[str, float] = {}
    for name, sel in selections.items():
        ab = eval_with_selection(model, gen, data_cfg, args, device, means, sel)
        ablated[name] = ab
        for hop in range(1, data_cfg.k_max + 1):
            d = acc_drop(base, ab, hop)
            if d is not None:
                summary[f'{name}_hop{hop}_drop'] = d
                summary[f'{name}_hop{hop}_ablated_acc'] = float(ab.get(f'hop{hop}_acc'))
                summary[f'base_hop{hop}_acc'] = float(base.get(f'hop{hop}_acc'))

    causal_ranking: Dict[str, Any] = {}
    if args.include_causal_ranking:
        single: List[Tuple[float, int, int]] = []
        for layer in range(model_cfg.n_layers):
            for head in range(model_cfg.n_heads):
                ab = eval_with_selection(model, gen, data_cfg, args, device, means, flat_head_selection(layer, head))
                d = acc_drop(base, ab, 2)
                single.append((float(d or 0.0), layer, head))
        single_sorted = sorted(single, reverse=True)
        causal_sel = selection_from_pairs([(l, h) for _d, l, h in single_sorted[:args.top_k]])
        selections['causal_drop'] = causal_sel
        ab = eval_with_selection(model, gen, data_cfg, args, device, means, causal_sel)
        ablated['causal_drop'] = ab
        for hop in range(1, data_cfg.k_max + 1):
            d = acc_drop(base, ab, hop)
            if d is not None:
                summary[f'causal_drop_hop{hop}_drop'] = d
                summary[f'causal_drop_hop{hop}_ablated_acc'] = float(ab.get(f'hop{hop}_acc'))
        causal_ranking = {
            'single_head_hop2_drops': [{'drop': d, 'layer': l, 'head': h} for d, l, h in single_sorted],
        }

    exclude_pairs: List[Tuple[int, int]] = []
    if not args.include_selected_in_random_pool:
        seen = set()
        for sel in selections.values():
            for p_ in flatten_selection(sel):
                if p_ not in seen:
                    exclude_pairs.append(p_); seen.add(p_)
    rng = random.Random(args.random_seed)
    random_draws: List[Dict[str, Any]] = []
    for draw_idx in range(args.num_random):
        sel = random_selection(rng, model_cfg.n_layers, model_cfg.n_heads, args.top_k, exclude_pairs)
        ab = eval_with_selection(model, gen, data_cfg, args, device, means, sel)
        random_draws.append({'draw_idx': draw_idx, 'selected_heads': sel, 'ablated': ab})
    summary.update(summarize_random(base, random_draws, data_cfg.k_max))
    for name in list(ablated.keys()):
        for hop in range(1, data_cfg.k_max + 1):
            sd = summary.get(f'{name}_hop{hop}_drop')
            rd = summary.get(f'random_hop{hop}_drop_mean')
            if sd is not None and rd is not None:
                summary[f'{name}_minus_random_hop{hop}_drop'] = float(sd) - float(rd)

    out = {
        'run_dir': str(run_dir),
        'top_k': args.top_k,
        'num_random': args.num_random,
        'random_seed': args.random_seed,
        'random_excludes_selected_union': not args.include_selected_in_random_pool,
        'score_names': list(scores.keys()),
        'scores': {k: v.tolist() for k, v in scores.items()},
        'selected_heads': selections,
        'selected_pairs': {k: flatten_selection(v) for k, v in selections.items()},
        'base': base,
        'ablated': ablated,
        'random_draws': random_draws,
        'summary': summary,
        **causal_ranking,
    }
    label = derive_label(run_dir)
    out_path = out_dir / f'twohop_score_ablation_{label}_top{args.top_k}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps({
        'run_dir': str(run_dir),
        'top_k': args.top_k,
        'selected_heads': {k: selection_str(v) for k, v in selections.items()},
        'base': base,
        'summary': summary,
        'out_path': str(out_path),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
