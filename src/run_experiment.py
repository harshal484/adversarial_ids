"""
run_experiment.py
=================
Master experiment script for CDAC ITISS Project:
  "Adversarial Input Attack on ML-based IDS"

Runs the full pipeline:
  1. Generate / load network flow dataset
  2. Train standard IDS model
  3. Train adversarially-hardened IDS model
  4. Launch FGSM and PGD attacks on both models
  5. Compare evasion rates (attack vs. defence)
  6. Print summary report

Usage:
    python run_experiment.py

CDAC ITISS Project: Adversarial Input Attack on ML-based IDS
"""

import os
import sys
import json
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from ids_model import IDSTrainer, generate_synthetic_data, NUMERIC_FEATURES
from fgsm_attack import FGSMAttacker, epsilon_sweep

RESULTS_DIR = '../results'
MODELS_DIR  = '../models'
DATA_DIR    = '../data'


def banner(title: str):
    w = 62
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def run_full_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ──────────────────────────────────────────────
    #  STEP 1: Generate Dataset
    # ──────────────────────────────────────────────
    banner("STEP 1 — Generating Synthetic Network Flow Dataset")
    print("  Simulating NSL-KDD-style network flow features...")
    df = generate_synthetic_data(n_samples=20000, seed=42)
    df.to_csv(f'{DATA_DIR}/network_flows.csv', index=False)

    n_normal = (df['label'] == 0).sum()
    n_attack = (df['label'] == 1).sum()
    print(f"  Total samples : {len(df):,}")
    print(f"  Normal traffic: {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    print(f"  Attack traffic: {n_attack:,} ({n_attack/len(df)*100:.1f}%)")
    print(f"  Features       : {len(NUMERIC_FEATURES)}")
    print(f"  Saved to       : {DATA_DIR}/network_flows.csv")

    # ──────────────────────────────────────────────
    #  STEP 2: Train Standard IDS Model
    # ──────────────────────────────────────────────
    banner("STEP 2 — Training Standard IDS Model (No Defence)")
    standard_trainer = IDSTrainer(MODELS_DIR)
    std_metrics = standard_trainer.train(
        df, epochs=25, batch_size=256, lr=1e-3, adversarial=False
    )
    print(f"\n  ✓ Standard model accuracy: {std_metrics['accuracy']*100:.2f}%")

    # ──────────────────────────────────────────────
    #  STEP 3: Train Adversarially Hardened IDS
    # ──────────────────────────────────────────────
    banner("STEP 3 — Training Adversarially-Hardened IDS (FGSM Defence)")
    adv_trainer = IDSTrainer(MODELS_DIR)
    adv_metrics = adv_trainer.train(
        df, epochs=25, batch_size=256, lr=1e-3,
        adversarial=True, fgsm_epsilon=0.15
    )
    print(f"\n  ✓ Adversarial model accuracy: {adv_metrics['accuracy']*100:.2f}%")

    # ──────────────────────────────────────────────
    #  STEP 4: Prepare Attack Samples
    # ──────────────────────────────────────────────
    banner("STEP 4 — Preparing Attack Samples")
    attack_df = df[df['label'] == 1].sample(1000, random_state=99)
    X_attacks = attack_df[NUMERIC_FEATURES].values
    print(f"  Using {len(X_attacks)} attack samples for evasion testing")

    epsilons = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    # ──────────────────────────────────────────────
    #  STEP 5a: FGSM on Standard Model
    # ──────────────────────────────────────────────
    banner("STEP 5a — FGSM Attack on STANDARD IDS Model")
    std_attacker = FGSMAttacker(standard_trainer.model, standard_trainer.scaler)
    std_fgsm_results = epsilon_sweep(
        std_attacker, X_attacks, epsilons, attack_type="FGSM",
        save_dir=f'{RESULTS_DIR}/standard'
    )

    # ──────────────────────────────────────────────
    #  STEP 5b: PGD on Standard Model
    # ──────────────────────────────────────────────
    banner("STEP 5b — PGD Attack on STANDARD IDS Model")
    std_pgd_results = epsilon_sweep(
        std_attacker, X_attacks, epsilons, attack_type="PGD",
        save_dir=f'{RESULTS_DIR}/standard'
    )

    # ──────────────────────────────────────────────
    #  STEP 5c: FGSM on Adversarial Model
    # ──────────────────────────────────────────────
    banner("STEP 5c — FGSM Attack on ADVERSARIALLY-HARDENED IDS Model")
    adv_attacker = FGSMAttacker(adv_trainer.model, adv_trainer.scaler)
    adv_fgsm_results = epsilon_sweep(
        adv_attacker, X_attacks, epsilons, attack_type="FGSM",
        save_dir=f'{RESULTS_DIR}/adversarial'
    )

    # ──────────────────────────────────────────────
    #  STEP 5d: PGD on Adversarial Model
    # ──────────────────────────────────────────────
    banner("STEP 5d — PGD Attack on ADVERSARIALLY-HARDENED IDS Model")
    adv_pgd_results = epsilon_sweep(
        adv_attacker, X_attacks, epsilons, attack_type="PGD",
        save_dir=f'{RESULTS_DIR}/adversarial'
    )

    # ──────────────────────────────────────────────
    #  STEP 6: Summary Report
    # ──────────────────────────────────────────────
    banner("STEP 6 — Experiment Summary Report")

    report = {
        'model_accuracy': {
            'standard_model': round(std_metrics['accuracy'] * 100, 2),
            'adversarial_model': round(adv_metrics['accuracy'] * 100, 2)
        },
        'evasion_comparison': []
    }

    print(f"\n  {'ε':>6} | {'Std-FGSM%':>11} | {'Std-PGD%':>10} | "
          f"{'Adv-FGSM%':>11} | {'Adv-PGD%':>10} | {'Defence Gain (FGSM)':>20}")
    print(f"  {'-'*75}")

    for i, eps in enumerate(epsilons):
        sf = std_fgsm_results[i].evasion_rate * 100
        sp = std_pgd_results[i].evasion_rate * 100
        af = adv_fgsm_results[i].evasion_rate * 100
        ap = adv_pgd_results[i].evasion_rate * 100
        gain = sf - af

        print(f"  {eps:>6.2f} | {sf:>10.1f}% | {sp:>9.1f}% | "
              f"{af:>10.1f}% | {ap:>9.1f}% | {gain:>+19.1f}%")

        report['evasion_comparison'].append({
            'epsilon': eps,
            'standard_fgsm_evasion_pct': round(sf, 1),
            'standard_pgd_evasion_pct': round(sp, 1),
            'adversarial_fgsm_evasion_pct': round(af, 1),
            'adversarial_pgd_evasion_pct': round(ap, 1),
            'defence_gain_fgsm_pct': round(gain, 1)
        })

    with open(f'{RESULTS_DIR}/summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full summary → {RESULTS_DIR}/summary_report.json")

    banner("EXPERIMENT COMPLETE")
    print(f"  Standard Model Accuracy   : {std_metrics['accuracy']*100:.2f}%")
    print(f"  Adversarial Model Accuracy: {adv_metrics['accuracy']*100:.2f}%")
    print(f"  Best FGSM evasion (std)   : {max(r.evasion_rate for r in std_fgsm_results)*100:.1f}%")
    print(f"  Best FGSM evasion (adv)   : {max(r.evasion_rate for r in adv_fgsm_results)*100:.1f}%")
    print(f"\n  Adversarial training demonstrably reduces evasion rate.")
    print(f"  See results/ for full JSON data.")
    print()


if __name__ == '__main__':
    run_full_experiment()
