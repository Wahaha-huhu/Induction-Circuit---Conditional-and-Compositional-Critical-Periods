#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    summary_dir = Path(args.summary_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    traj_path = summary_dir / "hop2_only_scale_trajectories.csv"
    per_path = summary_dir / "hop2_only_scale_per_run.csv"
    group_path = summary_dir / "hop2_only_scale_group_summary.csv"
    if not (traj_path.exists() and per_path.exists() and group_path.exists()):
        raise FileNotFoundError("Missing summary CSVs; run aggregate_hop2_only_scale_v0_21.py first")

    traj = pd.read_csv(traj_path)
    group = pd.read_csv(group_path)

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Curves by step-budget. With one seed these are usually nested repeats, but
    # the plot makes it easy to see whether any longer run eventually turns upward.
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    for (schedule, step_budget), d in traj.groupby(["schedule", "step_budget"]):
        d = d.sort_values("step")
        label = f"{schedule}, {int(step_budget/1000)}k"
        ax.plot(d["step"], 100 * d["hop2_acc"], linewidth=1.8, label=label)
    if "floor_hop2_acc" in traj.columns and traj["floor_hop2_acc"].notna().any():
        floor = traj.groupby("step", as_index=False)["floor_hop2_acc"].mean()
        ax.plot(floor["step"], 100 * floor["floor_hop2_acc"], linestyle=":", linewidth=1.2, label="Chance floor")
    ax.axhline(95, linestyle="--", linewidth=1, label="95% success")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"HOP$_2$ accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(r"HOP$_2$-only from-scratch scale sweep")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(out / "fig_hop2_only_scale_trajectories.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_hop2_only_scale_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    # Final/tail accuracy by budget.
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    g = group.sort_values("step_budget")
    ax.plot(g["step_budget"], 100 * g["mean_tail_hop2_acc"], marker="o", linewidth=2, label="Tail HOP_2 accuracy")
    if "sem_tail_hop2_acc" in g.columns and g["sem_tail_hop2_acc"].notna().any():
        ax.errorbar(g["step_budget"], 100 * g["mean_tail_hop2_acc"], yerr=100 * g["sem_tail_hop2_acc"].fillna(0), capsize=3, linestyle="none")
    ax.axhline(95, linestyle="--", linewidth=1, label="95% success")
    if "mean_tail_hop2_excess" in g.columns:
        # Chance floor is roughly tail acc - excess.
        floor = 100 * (g["mean_tail_hop2_acc"] - g["mean_tail_hop2_excess"])
        ax.plot(g["step_budget"], floor, linestyle=":", linewidth=1.2, label="Chance floor")
    ax.set_xlabel("Training budget (steps)")
    ax.set_ylabel(r"Tail HOP$_2$ accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(r"Does HOP$_2$-only training improve with scale?")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "fig_hop2_only_scale_tail_accuracy.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_hop2_only_scale_tail_accuracy.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
