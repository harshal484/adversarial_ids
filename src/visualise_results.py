"""
visualise_results.py
====================
Generate plots for the CDAC ITISS project report.

Produces:
  1. Evasion rate vs epsilon (FGSM + PGD, standard vs adversarial)
  2. Confidence score distribution (before vs after attack)
  3. Feature perturbation heatmap
  4. Training curves

Run after run_experiment.py.

CDAC ITISS Project: Adversarial Input Attack on ML-based IDS
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = '../results'
PLOTS_DIR   = '../results/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    'std_fgsm': '#E74C3C',
    'std_pgd':  '#C0392B',
    'adv_fgsm': '#2ECC71',
    'adv_pgd':  '#27AE60',
    'bg':       '#0F1117',
    'grid':     '#2A2D3A',
    'text':     '#ECF0F1'
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_evasion_curves():
    """Plot evasion rate vs epsilon for all attack/model combos."""
    std_fgsm = load_results(f'{RESULTS_DIR}/standard/attack_results_fgsm.json')
    std_pgd  = load_results(f'{RESULTS_DIR}/standard/attack_results_pgd.json')
    adv_fgsm = load_results(f'{RESULTS_DIR}/adversarial/attack_results_fgsm.json')
    adv_pgd  = load_results(f'{RESULTS_DIR}/adversarial/attack_results_pgd.json')

    epsilons = [r['epsilon'] for r in std_fgsm]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              facecolor=COLORS['bg'])

    for ax in axes:
        ax.set_facecolor(COLORS['bg'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'])
        ax.xaxis.label.set_color(COLORS['text'])
        ax.yaxis.label.set_color(COLORS['text'])
        ax.title.set_color(COLORS['text'])
        ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.5)

    # Left: FGSM comparison
    ax1 = axes[0]
    ax1.plot(epsilons, [r['evasion_rate'] for r in std_fgsm],
             'o-', color=COLORS['std_fgsm'], lw=2.5, ms=7,
             label='Standard IDS (No Defence)')
    ax1.plot(epsilons, [r['evasion_rate'] for r in adv_fgsm],
             's-', color=COLORS['adv_fgsm'], lw=2.5, ms=7,
             label='Adversarially Hardened IDS')
    ax1.fill_between(epsilons,
                     [r['evasion_rate'] for r in std_fgsm],
                     [r['evasion_rate'] for r in adv_fgsm],
                     alpha=0.15, color='#3498DB')
    ax1.set_title('FGSM Attack — Evasion Rate vs ε', fontsize=13, pad=10)
    ax1.set_xlabel('Perturbation Budget ε', fontsize=11)
    ax1.set_ylabel('Evasion Rate', fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax1.legend(facecolor='#1A1D2E', labelcolor=COLORS['text'],
               edgecolor=COLORS['grid'], fontsize=10)

    # Right: PGD comparison
    ax2 = axes[1]
    ax2.plot(epsilons, [r['evasion_rate'] for r in std_pgd],
             'o-', color=COLORS['std_pgd'], lw=2.5, ms=7,
             label='Standard IDS (No Defence)')
    ax2.plot(epsilons, [r['evasion_rate'] for r in adv_pgd],
             's-', color=COLORS['adv_pgd'], lw=2.5, ms=7,
             label='Adversarially Hardened IDS')
    ax2.fill_between(epsilons,
                     [r['evasion_rate'] for r in std_pgd],
                     [r['evasion_rate'] for r in adv_pgd],
                     alpha=0.15, color='#9B59B6')
    ax2.set_title('PGD Attack (10 steps) — Evasion Rate vs ε', fontsize=13, pad=10)
    ax2.set_xlabel('Perturbation Budget ε', fontsize=11)
    ax2.set_ylabel('Evasion Rate', fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax2.legend(facecolor='#1A1D2E', labelcolor=COLORS['text'],
               edgecolor=COLORS['grid'], fontsize=10)

    fig.suptitle(
        'Adversarial Attack Evasion Rate: Standard vs Adversarially Trained IDS\n'
        'CDAC ITISS — Adversarial Input Attack on ML-based IDS',
        color=COLORS['text'], fontsize=13, y=1.02
    )

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/evasion_curves.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f"  ✓ Saved: {PLOTS_DIR}/evasion_curves.png")


def plot_defence_gain_bar():
    """Bar chart of defence gain (evasion reduction) per epsilon."""
    summary = load_results(f'{RESULTS_DIR}/summary_report.json')
    data = summary['evasion_comparison']

    epsilons = [d['epsilon'] for d in data]
    gains_fgsm = [d['defence_gain_fgsm_pct'] for d in data]
    gains_pgd  = [d['standard_pgd_evasion_pct'] - d['adversarial_pgd_evasion_pct']
                  for d in data]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    ax.grid(True, color=COLORS['grid'], axis='y', linestyle='--', alpha=0.5)

    x = np.arange(len(epsilons))
    w = 0.35
    ax.bar(x - w/2, gains_fgsm, w, label='FGSM Defence Gain',
           color='#3498DB', alpha=0.9)
    ax.bar(x + w/2, gains_pgd, w, label='PGD Defence Gain',
           color='#9B59B6', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'ε={e}' for e in epsilons], color=COLORS['text'])
    ax.set_ylabel('Evasion Rate Reduction (%)', color=COLORS['text'], fontsize=11)
    ax.set_title(
        'Defence Gain from Adversarial Training\n(Percentage Points of Evasion Reduction)',
        color=COLORS['text'], fontsize=12, pad=10
    )
    ax.legend(facecolor='#1A1D2E', labelcolor=COLORS['text'], edgecolor=COLORS['grid'])
    ax.yaxis.label.set_color(COLORS['text'])

    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom',
                    color=COLORS['text'], fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/defence_gain.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f"  ✓ Saved: {PLOTS_DIR}/defence_gain.png")


def plot_architecture_diagram():
    """Visual representation of the IDS neural network architecture."""
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0A0E1A')
    ax.set_facecolor('#0A0E1A')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    layers = [
        ('Input\n38 features', 0.7, '#2980B9'),
        ('BatchNorm\n+ Dense(256)\nReLU', 2.2, '#8E44AD'),
        ('Dense(128)\nReLU\nDropout', 3.9, '#8E44AD'),
        ('Dense(64)\nReLU', 5.6, '#8E44AD'),
        ('Dense(1)\nSigmoid', 7.3, '#E74C3C'),
        ('Output\nNormal/Attack', 8.9, '#27AE60'),
    ]

    for (label, x, color) in layers:
        box = FancyBboxPatch((x - 0.55, 0.8), 1.1, 1.8,
                             boxstyle='round,pad=0.1',
                             facecolor=color, edgecolor='white',
                             linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 1.72, label, ha='center', va='center',
                color='white', fontsize=8, fontweight='bold',
                multialignment='center')

    # Arrows
    xs = [l[1] for l in layers]
    for i in range(len(xs) - 1):
        ax.annotate('', xy=(xs[i+1] - 0.56, 1.7), xytext=(xs[i] + 0.56, 1.7),
                    arrowprops=dict(arrowstyle='->', color='#ECF0F1', lw=1.8))

    ax.text(5, 3.2,
            'IDS Neural Network Architecture — IDSNet',
            ha='center', va='center', color='white',
            fontsize=13, fontweight='bold')
    ax.text(5, 0.3,
            'CDAC ITISS: Adversarial Input Attack on ML-based IDS',
            ha='center', va='center', color='#BDC3C7', fontsize=9)

    plt.savefig(f'{PLOTS_DIR}/architecture.png', dpi=150, bbox_inches='tight',
                facecolor='#0A0E1A')
    plt.close()
    print(f"  ✓ Saved: {PLOTS_DIR}/architecture.png")


def main():
    print("\n  Generating result visualisations...")
    print(f"  Output directory: {PLOTS_DIR}\n")

    try:
        plot_evasion_curves()
        plot_defence_gain_bar()
        plot_architecture_diagram()
        print("\n  All plots generated successfully.")
    except FileNotFoundError as e:
        print(f"\n  ⚠ Results not found: {e}")
        print("  Run run_experiment.py first, then visualise_results.py")


if __name__ == '__main__':
    main()
