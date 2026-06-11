from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass(frozen=True)
class DataConfig:
    """Frozen toy-generator defaults for the cascade tier.

    Token layout for each full sequence:
        [key, value, SEP] repeated M-1 times in random binding order,
        QUERY_MARKER, HOP_h, start_content_token, target

    The model input is the full sequence except the final target. The supervised
    loss is masked to the final target prediction only.

    Token pools:
        all:   content tokens [0, v_content)
        base:  first half of content tokens
        fresh: second half of content tokens

    The fresh/base split is used only for specificity controls; the main toy uses all.
    """

    v_content: int = 256
    chain_length: int = 16
    k_max: int = 2
    p_dynamic: float = 1.0
    p_multi: float = 0.5
    fixed_chain_seed: int = 12345
    # Special token ids are assigned after content tokens.
    sep_token: int = 256
    query_a_token: int = 257
    query_b_token: int = 258
    query_mem_token: int = 259
    hop_token_offset: int = 260  # HOP_h token id is hop_token_offset + h.

    @property
    def vocab_size(self) -> int:
        return self.hop_token_offset + self.k_max + 1

    @property
    def full_seq_len(self) -> int:
        # 3 tokens per binding, plus QUERY, HOP, start, target.
        return 3 * (self.chain_length - 1) + 4

    @property
    def input_seq_len(self) -> int:
        return self.full_seq_len - 1

    @property
    def query_start_input_pos(self) -> int:
        # The start content token is the final input token.
        return self.input_seq_len - 1


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class OptimConfig:
    peak_lr: float = 5e-4
    final_lr: float = 5e-6
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass(frozen=True)
class ScheduleConfig:
    kind: str = "warmup_cosine"  # warmup_cosine | warmup_constant | warmup_cyclic | warmup_cosine_then_rewarm_constant | warmup_cosine_then_rewarm_constant_reset_optim
    t_schedule: int = 20_000
    cycle_length: int = 2_000
    cycle_min_lr_frac: float = 0.1
    # C5b rewarm controls: use cosine before rewarm_step, then constant rewarm_lr.
    # The *_reset_optim variant also resets AdamW state at rewarm_step.
    # If rewarm_lr is None, optim.peak_lr is used.
    rewarm_step: Optional[int] = None
    rewarm_lr: Optional[float] = None


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 128
    max_steps: int = 20_000
    eval_interval: int = 50
    eval_batches: int = 16
    log_interval: int = 50
    device: str = "cuda"
    out_dir: str = "runs/debug"

    # Piecewise data schedule. Values are resolved by helper functions.
    p_dynamic_high: float = 1.0
    p_dynamic_low: float = 0.05
    p_multi_frozen: float = 0.5
    p_multi_before_intro: float = 0.0
    intro_step: Optional[int] = None
    dynamic_switch_step: Optional[int] = None

    # Query-marker schedule. Main experiments use A. C7 late routing uses A before
    # intro and B after intro, with both markers evaluated.
    query_marker: str = "A"
    query_marker_after_intro: Optional[str] = None
    eval_query_b: bool = False

    # Token-pool schedule. Main experiments use all. Fresh-single-hop controls can
    # train on base before intro and fresh after intro, with both pools evaluated.
    token_pool: str = "all"  # all | base | fresh
    token_pool_after_intro: Optional[str] = None
    eval_base_fresh: bool = False

    # Rank/consolidation logging is useful for final replications but can be slow.
    log_rank_metrics: bool = False


def to_json(obj) -> str:
    return json.dumps(asdict(obj), indent=2, sort_keys=True)
