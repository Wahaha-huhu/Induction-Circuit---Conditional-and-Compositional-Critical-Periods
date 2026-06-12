#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ARM_NAMES = {
    "s1_late_original", "s1_plateau_late", "s1_longcos_late", "s2_constant_late",
    "rewarm_late", "rewarm_reset_late", "fresh_hop1_s1", "fresh_hop1_s2",
}


def derive_arm(path: Path) -> str:
    for part in reversed(path.parts):
        if part in ARM_NAMES:
            return part
    return path.parent.name


def load_seed(run_dir: Path) -> Optional[int]:
    cfgp = run_dir / "config.json"
    if not cfgp.exists():
        return None
    try:
        cfg = json.loads(cfgp.read_text())
        return int(cfg.get("train", {}).get("seed"))
    except Exception:
        return None


def numeric(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate HOP_2 activation patching CSVs.")
    p.add_argument("--runs-dir", default="runs/behavioral_replication_v0_9")
    p.add_argument("--out-dir", default="runs/behavioral_replication_v0_9/hop2_activation_patching")
    args = p.parse_args()

    root = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    rows: List[Dict[str, Any]] = []
    for f in sorted(root.rglob("hop2_activation_patching.csv")):
        run_dir = f.parent
        arm = derive_arm(run_dir)
        seed = load_seed(run_dir)
        for r in read_csv(f):
            r = dict(r)
            r["run_dir"] = str(run_dir)
            r["arm"] = arm
            r["seed"] = seed
            rows.append(r)

    write_csv(out_dir / "activation_patching_all_sites.csv", rows)

    # Compact best residual/head summary per run.
    best_rows: List[Dict[str, Any]] = []
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_run.setdefault(str(r["run_dir"]), []).append(r)
    for rd, rs in by_run.items():
        base = next((r for r in rs if r.get("patch_kind") == "none"), None)
        residuals = [r for r in rs if r.get("patch_kind") == "residual"]
        heads = [r for r in rs if r.get("patch_kind") == "head"]
        def best(xs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not xs:
                return None
            return max(xs, key=lambda r: numeric(r.get("restoration_mean")) or -999)
        br = best(residuals)
        bh = best(heads)
        arm = rs[0].get("arm")
        seed = rs[0].get("seed")
        out = {"run_dir": rd, "arm": arm, "seed": seed}
        if base:
            out.update({f"baseline_{k}": v for k, v in base.items() if k.endswith("_mean")})
        if br:
            out.update({
                "best_residual_site": br.get("site"),
                "best_residual_layer": br.get("layer"),
                "best_residual_restoration": br.get("restoration_mean"),
                "best_residual_clean_answer_acc": br.get("patched_clean_answer_acc_mean"),
            })
        if bh:
            out.update({
                "best_head_site": bh.get("site"),
                "best_head_layer": bh.get("layer"),
                "best_head": bh.get("head"),
                "best_head_restoration": bh.get("restoration_mean"),
                "best_head_clean_answer_acc": bh.get("patched_clean_answer_acc_mean"),
            })
        best_rows.append(out)
    write_csv(out_dir / "activation_patching_best_by_run.csv", best_rows)

    report = out_dir / "activation_patching_report.md"
    report.write_text(
        "# HOP_2 activation patching report\n\n"
        "This is a narrow clean/corrupt activation patching analysis. It tests whether clean HOP_2 activations at the query position can restore the clean answer on a corrupted two-hop input.\n\n"
        "Use this as a compositional-representation probe, not as full causal scrubbing or proof of an exact head-to-head path.\n\n"
        "Outputs:\n"
        "- `activation_patching_all_sites.csv`\n"
        "- `activation_patching_best_by_run.csv`\n",
        encoding="utf-8",
    )
    print(f"aggregated {len(rows)} patching rows into {out_dir}")


if __name__ == "__main__":
    main()
