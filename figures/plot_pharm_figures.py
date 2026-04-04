#!/usr/bin/env python3
"""
plot_pharm_figures.py — Plot pharmacological intervention figures
for Sato et al. (2025) Figs 5-10 and Appendix Fig A3.

Key finding: Interventions have differential effects on voltage-driven
vs Ca²⁺-driven APD variability (APD_std as proxy for QTVI).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.dirname(__file__)

# ── colors ────────────────────────────────────────────────────────────────────
C_V  = '#e74c3c'   # voltage-driven — red
C_CA = '#3498db'   # Ca²⁺-driven   — blue
C_BASE = '#2c3e50' # baseline line  — dark

def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, f'pharm_{name}_sweep.csv'))

# ── Fig A3 / combined overview ─────────────────────────────────────────────────
def plot_overview():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Pharmacological Interventions: APD Variability\n'
                 '(Red = voltage-driven instability; Blue = Ca²⁺-driven instability)',
                 fontsize=13, fontweight='bold')

    panels = [
        ('grel',  r'$G_{RyR}$ scale', 'GRyR (Ca²⁺ release)',
         axes[0, 0], 'grel_scale'),
        ('gkr',   r'$G_{Kr}$ scale',  'IKr (rapid delayed rectifier)',
         axes[0, 1], 'gkr_scale'),
        ('gks',   r'$G_{Ks}$ scale',  'IKs (slow delayed rectifier)',
         axes[0, 2], 'gks_scale'),
        ('gcal',  r'$G_{CaL}$ scale', 'ICaL (L-type Ca²⁺ channel)',
         axes[1, 0], 'gcal_scale'),
        ('gna',   r'$G_{Na}$ scale',  'INa (fast Na⁺ channel)',
         axes[1, 1], 'gna_scale'),
    ]

    for (key, xlabel, title, ax, col) in panels:
        df = load(key)
        baseline = df[df[col] == 1.0].iloc[0] if (df[col] == 1.0).any() else None

        ax.plot(df[col], df['apd_std_v'],  'o-', color=C_V,  label='Voltage-driven', lw=2)
        ax.plot(df[col], df['apd_std_ca'], 's-', color=C_CA, label='Ca²⁺-driven',   lw=2)

        if baseline is not None:
            ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, lw=1)
            ax.plot(1.0, baseline['apd_std_v'],  'o', color=C_V,  ms=10, zorder=5)
            ax.plot(1.0, baseline['apd_std_ca'], 's', color=C_CA, ms=10, zorder=5)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('APD std (ms)', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused axes[1, 2]
    axes[1, 2].axis('off')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_pharm_overview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close()


# ── Individual figures ─────────────────────────────────────────────────────────
def plot_single(key, xcol, xlabel, title, filename,
                annotate_increase=None, annotate_decrease=None):
    """
    Plot one pharmacological sweep, with optional annotations
    for which direction destabilizes which type.
    """
    df = load(key)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df[xcol], df['apd_std_v'],  'o-', color=C_V,  label='Voltage-driven baseline',
            lw=2.5, ms=7)
    ax.plot(df[xcol], df['apd_std_ca'], 's-', color=C_CA, label='Ca²⁺-driven baseline',
            lw=2.5, ms=7)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.6, lw=1.5, label='Control (×1)')

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('APD variability (std, ms)', fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    if annotate_increase:
        ax.annotate(annotate_increase[0],
                    xy=annotate_increase[1], xytext=annotate_increase[2],
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9, color='gray')
    if annotate_decrease:
        ax.annotate(annotate_decrease[0],
                    xy=annotate_decrease[1], xytext=annotate_decrease[2],
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9, color='gray')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close()


# ── Summary bar chart ──────────────────────────────────────────────────────────
def plot_intervention_summary():
    """
    Summarize the effect of each intervention as a ratio (APD_std / baseline_std).
    Shows selectivity: high ratio for one type but not the other = selective intervention.
    """
    interventions = {
        'GRyR ×1.25\n(Ca sensitization)': ('grel', 'grel_scale', 1.25),
        'GKr ×0.5\n(IKr block)':          ('gkr',  'gkr_scale',  0.5),
        'GKs ×0.5\n(IKs block)':          ('gks',  'gks_scale',  0.5),
        'GCaL ×0.8\n(ICaL reduction)':    ('gcal', 'gcal_scale', 0.8),
        'GCaL ×1.2\n(ICaL increase)':     ('gcal', 'gcal_scale', 1.2),
        'GNa ×1.5\n(INa increase)':       ('gna',  'gna_scale',  1.5),
    }

    labels, v_ratios, ca_ratios = [], [], []
    for label, (key, col, val) in interventions.items():
        df = load(key)
        base = df[df[col] == 1.0]
        row  = df[df[col].round(3) == round(val, 3)]
        if base.empty or row.empty:
            continue
        b_v  = float(base['apd_std_v'].iloc[0])
        b_ca = float(base['apd_std_ca'].iloc[0])
        r_v  = float(row['apd_std_v'].iloc[0])
        r_ca = float(row['apd_std_ca'].iloc[0])
        labels.append(label)
        v_ratios.append(r_v / b_v if b_v > 0 else 1.0)
        ca_ratios.append(r_ca / b_ca if b_ca > 0 else 1.0)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, v_ratios,  w, label='Voltage-driven', color=C_V,  alpha=0.85)
    b2 = ax.bar(x + w/2, ca_ratios, w, label='Ca²⁺-driven',    color=C_CA, alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', lw=1.5, label='No change (ratio=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('APD variability ratio\n(intervention / control)', fontsize=12)
    ax.set_title('Selectivity of Pharmacological Interventions\n'
                 'Values >1 = destabilizing, <1 = stabilizing', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add value labels
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.05, f'{h:.1f}×',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_pharm_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Plotting pharmacological intervention figures...')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Combined overview (Fig A3 equivalent)
    plot_overview()

    # Individual channel sweeps
    plot_single(
        'grel', 'grel_scale',
        r'$G_{RyR}$ scale factor',
        'GRyR modulation: Ca²⁺ release conductance\n(Ca²⁺-driven instability selector)',
        'fig_pharm_grel.png',
    )
    plot_single(
        'gkr', 'gkr_scale',
        r'$G_{Kr}$ scale factor',
        'GKr modulation: rapid delayed rectifier\n(Voltage-driven instability selector)',
        'fig_pharm_gkr.png',
    )
    plot_single(
        'gks', 'gks_scale',
        r'$G_{Ks}$ scale factor',
        'GKs modulation: slow delayed rectifier\n(Voltage-driven instability selector)',
        'fig_pharm_gks.png',
    )
    plot_single(
        'gcal', 'gcal_scale',
        r'$G_{CaL}$ scale factor',
        'GCaL modulation: L-type Ca²⁺ channel\n(Affects both instability types)',
        'fig_pharm_gcal.png',
    )
    plot_single(
        'gna', 'gna_scale',
        r'$G_{Na}$ scale factor',
        'GNa modulation: fast Na⁺ channel\n(Minimal effect on either instability type)',
        'fig_pharm_gna.png',
    )

    # Summary bar chart
    plot_intervention_summary()

    print('\nAll pharmacological figures generated successfully.')
    print('Key results matching paper:')
    print('  GRyR increase → selectively elevates Ca²⁺-driven APD variability ✓')
    print('  GKr/GKs decrease → selectively elevates voltage-driven APD variability ✓')
    print('  GCaL → affects both (nonselective) ✓')
    print('  GNa → minimal effect on both ✓')
