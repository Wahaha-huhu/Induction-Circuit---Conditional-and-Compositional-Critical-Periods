#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError


def ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def arm_label(a: str) -> str:
    return {
        's1_cosine': 'S1 cosine',
        's2_constant': 'S2 constant',
        'rewarm_reset': 'Rewarm+reset',
    }.get(str(a), str(a))


def plot_pmulti(pm: pd.DataFrame, out: Path) -> None:
    if pm.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for arm, g in pm.groupby('arm'):
        g = g.sort_values('p_multi_grid')
        ax.errorbar(g['p_multi_grid'], 100*g['mean_tail_hop1_acc'], yerr=100*g.get('sd_tail_hop1_acc', 0), marker='o', capsize=3, label=f'{arm_label(arm)} HOP_1')
        ax.errorbar(g['p_multi_grid'], 100*g['mean_tail_hop2_acc'], yerr=100*g.get('sd_tail_hop2_acc', 0), marker='s', capsize=3, linestyle='--', label=f'{arm_label(arm)} HOP_2')
    ax.axhline(95, linestyle=':', linewidth=1.2)
    ax.set_xlabel(r'Mixture probability from start $p_{multi}$')
    ax.set_ylabel('Tail accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.03, 1.03)
    ax.set_title('Mixed-from-start feasibility sweep')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'fig_mixed_from_start_pmulti_tail_accuracy.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig_mixed_from_start_pmulti_tail_accuracy.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for arm, g in pm.groupby('arm'):
        g = g.sort_values('p_multi_grid')
        ax.plot(g['p_multi_grid'], 100*g['both_success_fraction'], marker='o', label=arm_label(arm))
    ax.set_xlabel(r'Mixture probability from start $p_{multi}$')
    ax.set_ylabel('Both-task success fraction (%)')
    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.03, 1.03)
    ax.set_title('Mixed-from-start both-task success')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'fig_mixed_from_start_both_success_fraction.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig_mixed_from_start_both_success_fraction.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_long(per: pd.DataFrame, out: Path) -> None:
    if per.empty:
        return
    # We cannot reconstruct full trajectories from summary CSV alone. Plot final/tail vs length.
    long = per[per['sweep_type'] == 'long_pmulti_0p5'].copy()
    if long.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for arm, g in long.groupby('arm'):
        g = g.sort_values('max_steps')
        ax.scatter(g['max_steps'], 100*g['tail_hop1_acc'], marker='o', label=f'{arm_label(arm)} HOP_1')
        ax.scatter(g['max_steps'], 100*g['tail_hop2_acc'], marker='s', label=f'{arm_label(arm)} HOP_2')
    ax.axhline(95, linestyle=':', linewidth=1.2)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Tail accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(r'Long mixed-from-start $p_{multi}=0.5$ runs')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'fig_long_pmulti_0p5_tail_accuracy.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig_long_pmulti_0p5_tail_accuracy.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def to_latex(pm: pd.DataFrame, long: pd.DataFrame, out: Path) -> None:
    tables = out / 'tables'
    ensure(tables)
    snippets = []
    if not pm.empty:
        df = pm[['arm', 'p_multi_grid', 'n_runs', 'n_both_success', 'both_success_fraction', 'mean_tail_hop1_acc', 'mean_tail_hop2_acc']].copy()
        for c in ['both_success_fraction', 'mean_tail_hop1_acc', 'mean_tail_hop2_acc']:
            df[c] = (100*df[c]).round(1)
        df.columns = ['Arm', r'$p_{multi}$', 'Runs', 'Both successes', r'Both success (\%)', r'HOP\_1 acc. (\%)', r'HOP\_2 acc. (\%)']
        tex = '\\begin{table}[t]\n\\centering\n' + df.to_latex(index=False, escape=False)
        tex += r'\\caption{Mixed-from-start feasibility sweep. HOP\_1 and HOP\_2 are sampled from the first training step, with no staged HOP\_1-only phase.}\n'
        tex += '\\label{tab:mixed_from_start_pmulti}\n\\end{table}\n'
        (tables / 'table_mixed_from_start_pmulti.tex').write_text(tex)
        snippets.append(tex)
    if not long.empty:
        df = long[['arm', 'long_steps_grid', 'n_runs', 'n_both_success', 'both_success_fraction', 'mean_tail_hop1_acc', 'mean_tail_hop2_acc']].copy()
        for c in ['both_success_fraction', 'mean_tail_hop1_acc', 'mean_tail_hop2_acc']:
            df[c] = (100*df[c]).round(1)
        df.columns = ['Arm', 'Steps', 'Runs', 'Both successes', r'Both success (\%)', r'HOP\_1 acc. (\%)', r'HOP\_2 acc. (\%)']
        tex = '\\begin{table}[t]\n\\centering\n' + df.to_latex(index=False, escape=False)
        tex += r'\\caption{Long mixed-from-start runs at $p_{multi}=0.5$. This checks whether simultaneous HOP\_1/HOP\_2 discovery emerges with a longer training budget.}\n'
        tex += '\\label{tab:long_mixed_from_start_p05}\n\\end{table}\n'
        (tables / 'table_long_mixed_from_start_p05.tex').write_text(tex)
        snippets.append(tex)
    (tables / 'all_mixed_from_start_v0_20_tables.tex').write_text('\n\n'.join(snippets))

    fig_snip = r'''
% v0.20 mixed-from-start feasibility figures.
\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/fig_mixed_from_start_pmulti_tail_accuracy.pdf}
    \caption{Mixed-from-start feasibility sweep. HOP\_1 and HOP\_2 examples are mixed from the first step, with no HOP\_1-only stage. This baseline asks whether the model can discover the prerequisite lookup and the composed two-hop rule simultaneously.}
    \label{fig:mixed_from_start_pmulti}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{figures/fig_long_pmulti_0p5_tail_accuracy.pdf}
    \caption{Long mixed-from-start runs at $p_{multi}=0.5$. This tests whether the failure of short mixed-from-start pilots is simply a budget issue.}
    \label{fig:long_mixed_from_start_p05}
\end{figure}
'''
    (out / 'v0_20_latex_snippet.tex').write_text(fig_snip.strip() + '\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary-dir', default='runs/mixed_from_start_sweeps_v0_20/summary')
    ap.add_argument('--out-dir', default='runs/mixed_from_start_sweeps_v0_20/figures')
    args = ap.parse_args()
    summary = Path(args.summary_dir)
    out = Path(args.out_dir)
    ensure(out)
    pm = safe_read(summary / 'pmulti_from_start_summary.csv')
    long = safe_read(summary / 'long_pmulti_0p5_summary.csv')
    per = safe_read(summary / 'mixed_from_start_v0_20_per_run.csv')
    plot_pmulti(pm, out)
    plot_long(per, out)
    to_latex(pm, long, out.parent)
    print(f'wrote figures and tables to {out.parent}')


if __name__ == '__main__':
    main()
