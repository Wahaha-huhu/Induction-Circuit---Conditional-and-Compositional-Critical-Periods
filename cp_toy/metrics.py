from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import Batch, ChainBatchGenerator
from .model import AblationMeans, HeadSelection, TinyTransformer


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy averaged over masked target positions."""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T), reduction="none").view(B, T)
    denom = loss_mask.sum().clamp_min(1.0)
    return (loss * loss_mask).sum() / denom


@torch.no_grad()
def batch_accuracy(logits: torch.Tensor, batch: Batch) -> float:
    pred = logits[:, -1, :].argmax(dim=-1)
    return (pred == batch.target).float().mean().item()


@torch.no_grad()
def evaluate(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device | str,
    p_dynamic: float = 1.0,
    p_multi: float = 0.0,
    force_hop: Optional[int] = None,
    query_marker: str = "A",
    shuffle_study_content: bool = False,
    ablate_heads: Optional[HeadSelection] = None,
    ablation_means: Optional[AblationMeans] = None,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []
    for _ in range(num_batches):
        batch = gen.batch(
            batch_size=batch_size,
            p_dynamic=p_dynamic,
            p_multi=p_multi,
            query_marker=query_marker,
            force_dynamic=True,
            force_hop=force_hop,
            shuffle_study_content=shuffle_study_content,
            device=device,
        )
        out = model(batch.input_ids, ablate_heads=ablate_heads, ablation_means=ablation_means)
        loss = masked_ce_loss(out["logits"], batch.labels, batch.loss_mask)
        losses.append(loss.item())
        accs.append(batch_accuracy(out["logits"], batch))
    return {"loss": float(sum(losses) / len(losses)), "accuracy": float(sum(accs) / len(accs))}


@torch.no_grad()
def evaluate_by_hop(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device | str,
    k_max: int,
    query_marker: str = "A",
    ablate_heads: Optional[HeadSelection] = None,
    ablation_means: Optional[AblationMeans] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for h in range(1, k_max + 1):
        res = evaluate(
            model,
            gen,
            batch_size=batch_size,
            num_batches=num_batches,
            device=device,
            p_dynamic=1.0,
            p_multi=0.0,
            force_hop=h,
            query_marker=query_marker,
            ablate_heads=ablate_heads,
            ablation_means=ablation_means,
        )
        out[f"hop{h}_acc"] = res["accuracy"]
        out[f"hop{h}_loss"] = res["loss"]
    return out


@torch.no_grad()
def content_shuffled_floor(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device | str,
    force_hop: int,
) -> float:
    res = evaluate(
        model,
        gen,
        batch_size=batch_size,
        num_batches=num_batches,
        device=device,
        p_dynamic=1.0,
        p_multi=0.0,
        force_hop=force_hop,
        shuffle_study_content=True,
    )
    return res["accuracy"]


@torch.no_grad()
def key_slot_lookup_scores(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Primary task-specific induction score over all layer-head pairs.

    Returns tensor [n_layers, n_heads]. It measures attention from the query start
    content token to the value slot in the binding [start, next, SEP].
    """
    model.eval()
    cfg = model.cfg
    accum = torch.zeros(cfg.n_layers, cfg.n_heads, device=device)
    count = 0
    for _ in range(num_batches):
        batch = gen.diagnostic_keyslot_batch(batch_size=batch_size, device=device)
        out = model(batch.input_ids, return_attn=True)
        qpos = batch.query_pos  # [B]
        vpos = batch.key_value_pos  # [B]
        b_idx = torch.arange(batch.input_ids.shape[0], device=device)
        for layer_idx, attn in enumerate(out["attns"]):  # B,H,T,T
            # Gather per example/head attention at qpos -> vpos.
            vals = attn[b_idx[:, None], torch.arange(cfg.n_heads, device=device)[None, :], qpos[:, None], vpos[:, None]]
            accum[layer_idx] += vals.sum(dim=0)
        count += batch.input_ids.shape[0]
    return (accum / max(1, count)).detach().cpu()


def top_heads_from_scores(scores: torch.Tensor, k: int = 1) -> HeadSelection:
    """Select top-k heads over all layer-head pairs, not per layer."""
    flat = scores.flatten()
    k = min(k, flat.numel())
    vals, idxs = torch.topk(flat, k=k)
    n_heads = scores.shape[1]
    selected: HeadSelection = {}
    for idx in idxs.tolist():
        layer = idx // n_heads
        head = idx % n_heads
        selected.setdefault(layer, []).append(head)
    return selected


@torch.no_grad()
def compute_global_head_means(
    model: TinyTransformer,
    gen: ChainBatchGenerator,
    batch_size: int,
    num_batches: int,
    device: torch.device | str,
) -> AblationMeans:
    """Compute one global mean output vector per layer/head over calibration data."""
    model.eval()
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for _ in range(num_batches):
        batch = gen.batch(batch_size=batch_size, p_dynamic=1.0, p_multi=0.5, force_dynamic=True, device=device)
        out = model(batch.input_ids, return_head_outputs=True)
        for layer_idx, head_out in enumerate(out["head_outputs"]):  # B,H,T,Dh
            # Mean over batch and positions, preserving head and d_head.
            layer_sum = head_out.sum(dim=(0, 2))
            if layer_idx not in sums:
                sums[layer_idx] = torch.zeros_like(layer_sum)
                counts[layer_idx] = 0
            sums[layer_idx] += layer_sum
            counts[layer_idx] += head_out.shape[0] * head_out.shape[2]
    return {layer: (sums[layer] / counts[layer]).detach().clone() for layer in sums}


def excess_over_floor(acc: float, floor: float) -> float:
    return max(0.0, acc - floor)
