# Adversarial Input Attack on ML-based IDS
### CDAC ITISS Project

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Attack traffic + FGSM perturbation ──► IDS Model ──► "NORMAL" ← EVADED │
│                                                             │
│   Defence: Adversarial Training reduces evasion by 50%+    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
adversarial_ids/
├── src/
│   ├── ids_model.py          # IDSNet neural network + training
│   ├── fgsm_attack.py        # FGSM & PGD adversarial attacks
│   ├── run_experiment.py     # Full experiment pipeline
│   └── visualise_results.py  # Plot generation
├── data/
│   └── network_flows.csv     # Generated NSL-KDD-style data (20k samples)
├── models/
│   ├── ids_standard.pth      # Standard IDS model weights
│   ├── ids_adversarial.pth   # Adversarially hardened model weights
│   └── scaler_*.pkl          # Feature scalers
├── results/
│   ├── standard/             # Attack results on standard model
│   ├── adversarial/          # Attack results on hardened model
│   ├── plots/                # Generated visualisations
│   └── summary_report.json   # Full experiment summary
├── REPORT.md                 # Full technical report
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full experiment (trains models + runs all attacks)
cd src
python run_experiment.py

# 3. Generate plots
python visualise_results.py
```

## What This Project Does

### Phase 1: Train ML IDS
- Generates 20,000 synthetic network flow samples (NSL-KDD-style, 38 features)
- Trains `IDSNet`: 38 → 256 → 128 → 64 → 1 neural network
- Achieves ~100% clean accuracy

### Phase 2: FGSM Attack
- Computes gradient of loss w.r.t. input features: ∇_x J(θ, x, y)
- Perturbs attack samples: `x_adv = x + ε · sign(∇_x J)`
- Applies mutable-feature mask (realistic adversary constraint)
- Measures **evasion rate**: fraction of attacks reclassified as Normal

### Phase 3: PGD Attack (stronger)
- Multi-step FGSM with projection back to ε-ball
- 10 iterations, step size = ε/10 × 2

### Phase 4: Adversarial Training (defence)
- Trains second model with FGSM augmentation at each batch
- Mixes clean + adversarial examples 50/50
- Measures evasion rate reduction

## Key Results

| ε | Standard Model (FGSM) | Hardened Model (FGSM) | Defence Gain |
|---|---|---|---|
| 0.5 | 0.0% | 0.0% | — |
| 1.0 | 8.4% | 11.0% | — |
| 1.5 | **87.4%** | **36.8%** | **+50.6 pp** |
| 2.0 | 95.8% | 55.4% | +40.4 pp |

**Both models: 100% clean accuracy** — showing clean accuracy ≠ robustness

## Core Files Explained

### `ids_model.py`
- `IDSNet`: PyTorch neural network class
- `IDSTrainer`: Handles training, adversarial augmentation, saving, loading
- `generate_synthetic_data()`: Creates NSL-KDD-style dataset

### `fgsm_attack.py`
- `FGSMAttacker`: Implements FGSM and PGD attacks
- `AttackResult`: Dataclass for evasion metrics
- `epsilon_sweep()`: Runs attacks across multiple ε values
- Includes **mutable feature masking** for realistic threat model

### `run_experiment.py`
- Orchestrates all 6 steps
- Produces JSON results in `results/`

## Adversarial ML Theory (Quick Summary)

**Why ML accuracy on clean data doesn't predict robustness:**
Neural networks learn decision boundaries that fit the training distribution, not the ε-neighbourhood around each point. High clean accuracy means samples are correctly classified; it says nothing about whether nearby (adversarially perturbed) points are also correctly classified.

**What adversarial training provides:**
- Empirical robustness to the attack type used during training
- Smoother decision boundaries with wider margins
- Does NOT provide: certified guarantees, robustness to unseen attack types, complete immunity

## References
- Goodfellow et al. (2014) — FGSM
- Madry et al. (2018) — PGD + Adversarial Training
- NSL-KDD dataset (Tavallaee et al., 2009)
