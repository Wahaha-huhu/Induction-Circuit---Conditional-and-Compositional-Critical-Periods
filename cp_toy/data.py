from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import DataConfig


@dataclass
class Batch:
    input_ids: torch.Tensor       # [B, T]
    labels: torch.Tensor          # [B, T]
    loss_mask: torch.Tensor       # [B, T], supervised only on final target label
    target: torch.Tensor          # [B]
    hop: torch.Tensor             # [B]
    is_dynamic: torch.Tensor      # [B]
    key_value_pos: torch.Tensor   # [B], value position for the first hop key-slot lookup (legacy alias)
    query_pos: torch.Tensor       # [B], query start content position in input_ids
    distance_max: torch.Tensor    # [B]
    start_token: torch.Tensor     # [B]
    first_hop_key_pos: torch.Tensor    # [B], key position for first edge A->B
    first_hop_value_pos: torch.Tensor  # [B], value position B for first edge A->B
    second_hop_key_pos: torch.Tensor   # [B], key position for second edge B->C; -1 for HOP_1
    second_hop_value_pos: torch.Tensor # [B], value position C for second edge B->C; first value for HOP_1
    intermediate_token: torch.Tensor   # [B], B for HOP_2, target for HOP_1


class ChainBatchGenerator:
    """Task-structured chain generator.

    The generator supports dynamic fresh chains and a fixed memorisable chain.
    It always returns target-only next-token supervision.

    Important diagnostic detail:
    The key-slot lookup score uses `key_value_pos`: the value slot in the binding
    [c_s, c_{s+1}, SEP] for the first hop. This avoids the vanilla induction
    ambiguity caused by interior chain tokens also appearing as values.

    Token pools:
      - all: sample from the full content vocabulary.
      - base: sample from the first half of the content vocabulary.
      - fresh: sample from the second half of the content vocabulary.

    The base/fresh split supports the late fresh-single-hop specificity control.
    """

    def __init__(self, cfg: DataConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        fixed_rng = np.random.default_rng(cfg.fixed_chain_seed)
        self.fixed_chains = {
            pool: self._choice_from_pool(fixed_rng, pool, cfg.chain_length)
            for pool in ("all", "base", "fresh")
        }

    def _pool_tokens(self, token_pool: str) -> np.ndarray:
        token_pool = token_pool.lower()
        if token_pool == "all":
            lo, hi = 0, self.cfg.v_content
        elif token_pool == "base":
            lo, hi = 0, self.cfg.v_content // 2
        elif token_pool == "fresh":
            lo, hi = self.cfg.v_content // 2, self.cfg.v_content
        else:
            raise ValueError(f"unknown token_pool {token_pool}; expected all|base|fresh")
        if hi - lo < self.cfg.chain_length:
            raise ValueError(
                f"token_pool={token_pool} has only {hi-lo} tokens, less than chain_length={self.cfg.chain_length}"
            )
        return np.arange(lo, hi, dtype=np.int64)

    def _choice_from_pool(self, rng: np.random.Generator, token_pool: str, size: int) -> np.ndarray:
        pool = self._pool_tokens(token_pool)
        return rng.choice(pool, size=size, replace=False).astype(np.int64)

    def _hop_token(self, h: int) -> int:
        return self.cfg.hop_token_offset + h

    def _query_token(self, marker: str) -> int:
        marker = marker.upper()
        if marker == "A":
            return self.cfg.query_a_token
        if marker == "B":
            return self.cfg.query_b_token
        if marker == "MEM":
            return self.cfg.query_mem_token
        raise ValueError(f"unknown query marker: {marker}")

    def _sample_chain(self, dynamic: bool, token_pool: str = "all") -> np.ndarray:
        token_pool = token_pool.lower()
        if dynamic:
            return self._choice_from_pool(self.rng, token_pool, self.cfg.chain_length)
        return self.fixed_chains[token_pool].copy()

    def _sample_hop(self, p_multi: float) -> int:
        if self.cfg.k_max <= 1:
            return 1
        if self.rng.random() < p_multi:
            return int(self.rng.integers(2, self.cfg.k_max + 1))
        return 1

    def _make_one(
        self,
        p_dynamic: float,
        p_multi: float,
        query_marker: str = "A",
        token_pool: str = "all",
        force_dynamic: Optional[bool] = None,
        force_hop: Optional[int] = None,
        shuffle_study_content: bool = False,
    ) -> Tuple[List[int], Dict[str, int]]:
        dynamic = bool(self.rng.random() < p_dynamic) if force_dynamic is None else bool(force_dynamic)
        chain = self._sample_chain(dynamic, token_pool=token_pool)
        h = self._sample_hop(p_multi) if force_hop is None else int(force_hop)
        if h < 1 or h > self.cfg.k_max:
            raise ValueError(f"hop {h} outside [1, {self.cfg.k_max}]")

        # Start index must have h following links.
        s = int(self.rng.integers(0, self.cfg.chain_length - h))
        start = int(chain[s])
        target = int(chain[s + h])

        # Randomise binding order to block positional shortcuts.
        binding_indices = list(range(self.cfg.chain_length - 1))
        self.rng.shuffle(binding_indices)

        seq: List[int] = []
        first_hop_value_pos: Optional[int] = None
        first_hop_key_pos: Optional[int] = None
        second_hop_value_pos: Optional[int] = None
        second_hop_key_pos: Optional[int] = None
        all_hop_value_positions: List[int] = []

        # Optional shuffled-content floor: corrupt the study content but keep the
        # query and target from the original chain.
        if shuffle_study_content:
            shuffled_chain = chain.copy()
            self.rng.shuffle(shuffled_chain)
            study_chain = shuffled_chain
        else:
            study_chain = chain

        for i in binding_indices:
            key = int(study_chain[i])
            value = int(study_chain[i + 1])
            key_pos = len(seq)
            value_pos = len(seq) + 1
            seq.extend([key, value, self.cfg.sep_token])

            if not shuffle_study_content:
                if i == s:
                    first_hop_key_pos = key_pos
                    first_hop_value_pos = value_pos
                if h >= 2 and i == s + 1:
                    second_hop_key_pos = key_pos
                    second_hop_value_pos = value_pos
                if s <= i < s + h:
                    all_hop_value_positions.append(value_pos)

        # Query uses the content start token, not a generic start symbol.
        query_token = self._query_token(query_marker)
        seq.extend([query_token, self._hop_token(h), start, target])

        if first_hop_value_pos is None:
            # This should only happen for shuffled-study diagnostics; fall back to
            # zero rather than failing batch construction.
            first_hop_value_pos = 0
            first_hop_key_pos = 0
        if second_hop_value_pos is None:
            # For HOP_1 diagnostics there is no second edge.  Use the first value
            # as a harmless placeholder and set the second key to -1.
            second_hop_value_pos = int(first_hop_value_pos)
            second_hop_key_pos = -1
        query_pos = self.cfg.query_start_input_pos
        if all_hop_value_positions:
            distance_max = max(abs(query_pos - p) for p in all_hop_value_positions)
        else:
            distance_max = abs(query_pos - int(first_hop_value_pos))

        meta = {
            "target": target,
            "hop": h,
            "is_dynamic": int(dynamic),
            "key_value_pos": int(first_hop_value_pos),
            "query_pos": int(query_pos),
            "distance_max": int(distance_max),
            "start_token": start,
            "first_hop_key_pos": int(first_hop_key_pos),
            "first_hop_value_pos": int(first_hop_value_pos),
            "second_hop_key_pos": int(second_hop_key_pos),
            "second_hop_value_pos": int(second_hop_value_pos),
            "intermediate_token": int(chain[s + 1]) if h >= 2 else target,
        }
        return seq, meta

    def batch(
        self,
        batch_size: int,
        p_dynamic: Optional[float] = None,
        p_multi: Optional[float] = None,
        query_marker: str = "A",
        token_pool: str = "all",
        force_dynamic: Optional[bool] = None,
        force_hop: Optional[int] = None,
        shuffle_study_content: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> Batch:
        p_dynamic = self.cfg.p_dynamic if p_dynamic is None else float(p_dynamic)
        p_multi = self.cfg.p_multi if p_multi is None else float(p_multi)

        seqs: List[List[int]] = []
        metas: List[Dict[str, int]] = []
        for _ in range(batch_size):
            seq, meta = self._make_one(
                p_dynamic=p_dynamic,
                p_multi=p_multi,
                query_marker=query_marker,
                token_pool=token_pool,
                force_dynamic=force_dynamic,
                force_hop=force_hop,
                shuffle_study_content=shuffle_study_content,
            )
            seqs.append(seq)
            metas.append(meta)

        full = torch.tensor(seqs, dtype=torch.long, device=device)
        input_ids = full[:, :-1]
        labels = full[:, 1:]
        loss_mask = torch.zeros_like(labels, dtype=torch.float32)
        loss_mask[:, -1] = 1.0

        def meta_tensor(key: str) -> torch.Tensor:
            return torch.tensor([m[key] for m in metas], dtype=torch.long, device=device)

        return Batch(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            target=meta_tensor("target"),
            hop=meta_tensor("hop"),
            is_dynamic=meta_tensor("is_dynamic"),
            key_value_pos=meta_tensor("key_value_pos"),
            query_pos=meta_tensor("query_pos"),
            distance_max=meta_tensor("distance_max"),
            start_token=meta_tensor("start_token"),
            first_hop_key_pos=meta_tensor("first_hop_key_pos"),
            first_hop_value_pos=meta_tensor("first_hop_value_pos"),
            second_hop_key_pos=meta_tensor("second_hop_key_pos"),
            second_hop_value_pos=meta_tensor("second_hop_value_pos"),
            intermediate_token=meta_tensor("intermediate_token"),
        )

    def diagnostic_keyslot_batch(self, batch_size: int, device=None, token_pool: str = "all") -> Batch:
        """Held-out task-structured single-hop dynamic diagnostic set.

        This is the primary substrate for the key-slot lookup score. It is not the
        vanilla repeated-token E4 set.
        """
        return self.batch(
            batch_size=batch_size,
            p_dynamic=1.0,
            p_multi=0.0,
            query_marker="A",
            token_pool=token_pool,
            force_dynamic=True,
            force_hop=1,
            shuffle_study_content=False,
            device=device,
        )
