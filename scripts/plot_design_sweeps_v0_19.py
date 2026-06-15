#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def as_pct(x):
    if pd.isna(x):
        return x
    return 100.0 * float(x)


def label_arm(a: str) -> str:
    return {
        "s1_cosine": "S1 cosine",
        "s2_constant": "S2 constant",
        "rewarm_reset": "Rewarm+reset",
        "rewarm": "Rewarm",
    }.get(str(a), str(a))


def plot_line_summary(df: pd.DataFrame, x: str, out_prefix: Path, title: str, xlabel: str, footer: Optional[str] = None) -> None:
    if df.empty or x not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for arm, g in df.groupby("arm", dropna=False):
        gg = g.sort_values(x)
        y = gg["mean_tail_hop2_acc"].map(as_pct)
        yerr = gg.get("sd_tail_hop2_acc", pd.Series([math.nan] * len(gg))).map(as_pct)
        ax.plot(gg[x], y, marker="o", label=label_arm(arm))
        if yerr.notna().any():
            ax.fill_between(gg[x], y - yerr, y + yerr, alpha=0.12)
    ax.axhline(95, linestyle="--", linewidth=1.1, label="95% success threshold")
    ax.set_ylim(-2, 103)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final HOP$_2$ accuracy (%)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    if footer:
        fig.text(0.01, 0.01, footer, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_success_bar(df: pd.DataFrame, x: str, out_prefix: Path, title: str, xlabel: str) -> None:
    if df.empty or x not in df.columns:
        return
    arms = list(dict.fromkeys(df["arm"].astype(str).tolist()))
    xs = sorted(df[x].dropna().unique())
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    width = 0.8 / max(1, len(arms))
    centers = list(range(len(xs)))
    for i, arm in enumerate(arms):
        vals = []
        for xv in xs:
            row = df[(df["arm"].astype(str) == arm) & (df[x] == xv)]
            vals.append(float(row["success_fraction"].iloc[0]) * 100 if len(row) else math.nan)
        pos = [c - 0.4 + width * (i + 0.5) for c in centers]
        ax.bar(pos, vals, width=width, label=label_arm(arm))
    ax.set_xticks(centers)
    ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v) for v in xs])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success fraction (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def to_latex_table(df: pd.DataFrame, cols: List[str], path: Path, caption: str, label: str) -> None:
    if df.empty:
        return
    show = df.copy()
    for c in ["mean_tail_hop2_acc", "sd_tail_hop2_acc", "mean_tail_hop2_excess", "success_fraction", "mean_intro_hop1_acc"]:
        if c in show.columns:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{100*float(v):.1f}")
    for c in ["w_post", "intro_step_grid", "n_runs", "n_success"]:
        if c in show.columns:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    if "p_multi_grid" in show.columns:
        show["p_multi_grid"] = show["p_multi_grid"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
    if "arm" in show.columns:
        show["arm"] = show["arm"].map(label_arm)
    rename = {
        "arm": "Arm",
        "w_post": "$W_{post}$",
        "intro_step_grid": "$\\tau_{intro}$",
        "p_multi_grid": "$p_{multi}$",
        "n_runs": "$n$",
        "n_success": "Successes",
        "success_fraction": "Success (\\%)",
        "mean_tail_hop2_acc": "HOP\\_2 acc. (\\%)",
        "mean_tail_hop2_excess": "HOP\\_2 excess (pp)",
        "mean_intro_hop1_acc": "Intro HOP\\_1 acc. (\\%)",
    }
    show = show[cols].rename(columns=rename)
    tex = show.to_latex(index=False, escape=False)
    wrapped = "\\begin{table}[t]\n\\centering\n" + tex + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    path.write_text(wrapped, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="runs/design_sweeps_v0_19/summary")
    ap.add_argument("--out-dir", default="runs/design_sweeps_v0_19/figures")
    args = ap.parse_args()
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)

    def read(name: str) -> pd.DataFrame:
        p = summary_dir / name
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except EmptyDataError:
            return pd.DataFrame()

    wpost = read("wpost_calibration_summary.csv")
    intro = read("intro_step_sweep_summary.csv")
    mix = read("mixture_sensitivity_summary.csv")

    plot_line_summary(
        wpost,
        "w_post",
        out_dir / "fig0b_wpost_calibration",
        "Calibrating the fixed post-introduction HOP$_2$ budget",
        "Post-introduction training steps $W_{post}$",
        "Choose the smallest budget where high-update schedules reliably learn HOP$_2$.",
    )
    plot_success_bar(
        wpost,
        "w_post",
        out_dir / "fig0b_wpost_calibration_success_fraction",
        "HOP$_2$ success fraction across post-introduction budgets",
        "Post-introduction training steps $W_{post}$",
    )
    plot_line_summary(
        intro,
        "intro_step_grid",
        out_dir / "fig0c_intro_step_vs_final_hop2_accuracy",
        "Introduction step vs final HOP$_2$ accuracy at fixed post-introduction budget",
        "HOP$_2$ introduction step $\\tau_{intro}$",
        "All points use the same post-introduction budget; differences reflect acquirability under the schedule.",
    )
    plot_success_bar(
        intro,
        "intro_step_grid",
        out_dir / "fig0c_intro_step_success_fraction",
        "HOP$_2$ success fraction across introduction steps",
        "HOP$_2$ introduction step $\\tau_{intro}$",
    )
    plot_line_summary(
        mix,
        "p_multi_grid",
        out_dir / "fig0d_mixture_sensitivity",
        "Mixture sensitivity at late HOP$_2$ introduction",
        "Post-introduction HOP$_2$ sampling probability $p_{multi}$",
        "Before introduction, $p_{multi}=0$ in all cases; the sweep tests robustness to the post-introduction mixture ratio.",
    )
    plot_success_bar(
        mix,
        "p_multi_grid",
        out_dir / "fig0d_mixture_success_fraction",
        "HOP$_2$ success fraction across mixture ratios",
        "Post-introduction HOP$_2$ sampling probability $p_{multi}$",
    )

    to_latex_table(
        wpost,
        ["arm", "w_post", "n_runs", "n_success", "success_fraction", "mean_tail_hop2_acc", "mean_tail_hop2_excess"],
        table_dir / "table_wpost_calibration.tex",
        "Post-introduction budget calibration. Values report tail HOP\\_2 accuracy and success fraction after HOP\\_2 introduction under high-update reference schedules.",
        "tab:wpost_calibration",
    )
    to_latex_table(
        intro,
        ["arm", "intro_step_grid", "n_runs", "n_success", "success_fraction", "mean_intro_hop1_acc", "mean_tail_hop2_acc", "mean_tail_hop2_excess"],
        table_dir / "table_intro_step_sweep.tex",
        "Introduction-step sweep with fixed post-introduction budget. The HOP\\_2 budget after introduction is held fixed, so changes with $\\tau_{intro}$ reflect schedule-dependent acquirability rather than fewer examples.",
        "tab:intro_step_sweep",
    )
    to_latex_table(
        mix,
        ["arm", "p_multi_grid", "n_runs", "n_success", "success_fraction", "mean_tail_hop2_acc", "mean_tail_hop2_excess"],
        table_dir / "table_mixture_sensitivity.tex",
        "Mixture-ratio sensitivity at late introduction. HOP\\_2 is absent before introduction in all conditions; after introduction, $p_{multi}$ is swept.",
        "tab:mixture_sensitivity",
    )

    snippet = r'''
% v0.19 design-validation figures and tables.
\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/fig0b_wpost_calibration.pdf}
    \caption{Calibration of the fixed post-introduction HOP\_2 budget. We choose $W_{post}$ as the smallest post-introduction budget under which high-update reference schedules reliably acquire HOP\_2. This makes the later introduction-step sweep conservative: HOP\_2 is learnable in principle within the fixed budget.}
    \label{fig:wpost_calibration}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/fig0c_intro_step_vs_final_hop2_accuracy.pdf}
    \caption{HOP\_2 introduction step versus final HOP\_2 accuracy with a fixed post-introduction budget. Because $W_{post}$ is held fixed, the sweep separates schedule-dependent acquirability from the trivial effect of receiving fewer HOP\_2 examples.}
    \label{fig:intro_step_sweep}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/fig0d_mixture_sensitivity.pdf}
    \caption{Mixture-ratio sensitivity. Before introduction, HOP\_2 is absent in all conditions. After introduction, HOP\_2 is sampled with probability $p_{multi}$. This sweep checks that the late-HOP\_2 result is not an artefact of choosing $p_{multi}=0.5$.}
    \label{fig:mixture_sensitivity}
\end{figure}
'''
    (out_dir / "v0_19_latex_snippet.tex").write_text(snippet.strip() + "\n", encoding="utf-8")
    print(f"wrote figures and tables to {out_dir}")


if __name__ == "__main__":
    main()
