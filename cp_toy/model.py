from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


AblationMeans = Dict[int, torch.Tensor]  # layer -> [n_heads, d_head]
HeadSelection = Dict[int, List[int]]     # layer -> list of head indices


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        return_attn: bool = False,
        return_head_outputs: bool = False,
        ablate_heads: Optional[HeadSelection] = None,
        ablation_means: Optional[AblationMeans] = None,
        patch_head_outputs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # B,H,T,D
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        mask = self.causal_mask[:T, :T]
        att = att.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
        att_prob = F.softmax(att, dim=-1)
        att_prob = self.dropout(att_prob)
        head_out = att_prob @ v  # B,H,T,Dh

        # Mean ablation: replace selected heads by one global mean vector per head.
        if ablate_heads and layer_idx in ablate_heads:
            for h in ablate_heads[layer_idx]:
                if ablation_means is not None and layer_idx in ablation_means:
                    mean_vec = ablation_means[layer_idx][h].to(head_out.device, head_out.dtype)
                    head_out[:, h, :, :] = mean_vec.view(1, 1, self.d_head)
                else:
                    head_out[:, h, :, :] = 0.0

        # Optional activation patching for mechanistic diagnostics.  A patch spec
        # is a dict with:
        #   source: clean head outputs [B,H,T,Dh]
        #   heads: optional list of heads to patch; default all heads
        #   positions: optional int/list/tensor positions to patch; default all positions
        # This is intentionally disabled by default and used only by analysis scripts.
        if patch_head_outputs and layer_idx in patch_head_outputs:
            spec = patch_head_outputs[layer_idx]
            source = spec["source"].to(head_out.device, head_out.dtype)
            heads = spec.get("heads", None)
            positions = spec.get("positions", None)
            if heads is None:
                heads = list(range(self.n_heads))
            if isinstance(heads, int):
                heads = [heads]
            if positions is None:
                head_out[:, heads, :, :] = source[:, heads, :, :]
            else:
                if isinstance(positions, int):
                    head_out[:, heads, positions, :] = source[:, heads, positions, :]
                else:
                    pos_t = torch.as_tensor(positions, device=head_out.device, dtype=torch.long)
                    if pos_t.ndim == 0:
                        head_out[:, heads, int(pos_t.item()), :] = source[:, heads, int(pos_t.item()), :]
                    elif pos_t.ndim == 1 and pos_t.numel() == B:
                        b_idx = torch.arange(B, device=head_out.device)
                        for h in heads:
                            head_out[b_idx, h, pos_t, :] = source[b_idx, h, pos_t, :]
                    else:
                        head_out[:, heads, pos_t, :] = source[:, heads, pos_t, :]

        y = head_out.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)
        return y, (att_prob if return_attn else None), (head_out if return_head_outputs else None)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        return_attn: bool = False,
        return_head_outputs: bool = False,
        ablate_heads: Optional[HeadSelection] = None,
        ablation_means: Optional[AblationMeans] = None,
        patch_head_outputs: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        attn_out, attn, head_out = self.attn(
            self.ln1(x),
            layer_idx=layer_idx,
            return_attn=return_attn,
            return_head_outputs=return_head_outputs,
            ablate_heads=ablate_heads,
            ablation_means=ablation_means,
            patch_head_outputs=patch_head_outputs,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn, head_out


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attn: bool = False,
        return_head_outputs: bool = False,
        ablate_heads: Optional[HeadSelection] = None,
        ablation_means: Optional[AblationMeans] = None,
        return_residuals: bool = False,
        patch_residuals: Optional[Dict[int, Dict[str, Any]]] = None,
        patch_head_outputs: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        B, T = input_ids.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"input length {T} exceeds model seq_len {self.cfg.seq_len}")
        pos = torch.arange(T, device=input_ids.device).view(1, T)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        attns: List[torch.Tensor] = []
        head_outputs: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            x, attn, head_out = block(
                x,
                layer_idx=layer_idx,
                return_attn=return_attn,
                return_head_outputs=return_head_outputs,
                ablate_heads=ablate_heads,
                ablation_means=ablation_means,
                patch_head_outputs=patch_head_outputs,
            )
            # Optional residual-stream patch after this block.  Spec:
            #   source: clean residual [B,T,C]
            #   positions: optional int/list/tensor; default all positions
            if patch_residuals and layer_idx in patch_residuals:
                spec = patch_residuals[layer_idx]
                source = spec["source"].to(x.device, x.dtype)
                positions = spec.get("positions", None)
                if positions is None:
                    x = source.clone()
                elif isinstance(positions, int):
                    x[:, positions, :] = source[:, positions, :]
                else:
                    pos_t = torch.as_tensor(positions, device=x.device, dtype=torch.long)
                    if pos_t.ndim == 0:
                        x[:, int(pos_t.item()), :] = source[:, int(pos_t.item()), :]
                    elif pos_t.ndim == 1 and pos_t.numel() == B:
                        b_idx = torch.arange(B, device=x.device)
                        x[b_idx, pos_t, :] = source[b_idx, pos_t, :]
                    else:
                        x[:, pos_t, :] = source[:, pos_t, :]
            if return_residuals:
                residuals.append(x.detach())
            if return_attn:
                attns.append(attn)
            if return_head_outputs:
                head_outputs.append(head_out)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if return_attn:
            out["attns"] = attns
        if return_head_outputs:
            out["head_outputs"] = head_outputs
        if return_residuals:
            out["residuals"] = residuals
        return out


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
