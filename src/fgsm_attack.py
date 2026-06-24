"""
fgsm_attack.py
==============
Fast Gradient Sign Method (FGSM) adversarial attack on network flow features.

The FGSM attack was introduced by Goodfellow et al. (2014):
  x_adv = x + ε · sign(∇_x J(θ, x, y))

In the IDS context:
  - x     = scaled network flow feature vector (attacker-controlled fields only)
  - y     = attack label (1)
  - J     = BCE loss
  - ε     = perturbation budget
  - Goal  = maximise loss → make the model classify attack as normal

CDAC ITISS Project: Adversarial Input Attack on ML-based IDS
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
import os

# ─────────────────────────────────────────────
#  ATTACKER-CONTROLLABLE FEATURE INDICES
#  (Features an adversary can realistically modify
#   without breaking packet semantics)
# ─────────────────────────────────────────────
MUTABLE_FEATURE_NAMES = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
    'hot', 'num_failed_logins', 'logged_in',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate'
]

NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

MUTABLE_INDICES = [NUMERIC_FEATURES.index(f) for f in MUTABLE_FEATURE_NAMES]


@dataclass
class AttackResult:
    """Stores results from a single adversarial attack run."""
    epsilon: float
    n_original: int
    n_evaded: int
    evasion_rate: float
    avg_perturbation_l2: float
    avg_perturbation_linf: float
    confidence_drop: float  # Mean drop in attack probability
    iterations: int = 1
    attack_type: str = "FGSM"
    notes: str = ""

    def to_dict(self):
        return {
            'epsilon': self.epsilon,
            'n_original': self.n_original,
            'n_evaded': self.n_evaded,
            'evasion_rate': round(self.evasion_rate * 100, 2),
            'avg_perturbation_l2': round(self.avg_perturbation_l2, 4),
            'avg_perturbation_linf': round(self.avg_perturbation_linf, 4),
            'confidence_drop': round(self.confidence_drop, 4),
            'iterations': self.iterations,
            'attack_type': self.attack_type,
            'notes': self.notes
        }


# ─────────────────────────────────────────────
#  FGSM ATTACKER CLASS
# ─────────────────────────────────────────────
class FGSMAttacker:
    """
    Crafts adversarial network flow examples using FGSM.

    The attacker operates in the scaled feature space (what the model sees),
    but applies perturbations only to mutable features — those that an
    adversary can control in real traffic without breaking functionality.

    White-box assumption: attacker knows the model weights (worst case).
    """

    def __init__(self, model: nn.Module, scaler, epsilon: float = 0.1):
        self.model = model
        self.scaler = scaler
        self.epsilon = epsilon
        self.criterion = nn.BCELoss()
        self.model.eval()

    def _mask_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero-out gradients for immutable features (realistic constraint)."""
        mask = torch.zeros_like(grad)
        mask[:, MUTABLE_INDICES] = 1.0
        return grad * mask

    def attack_fgsm(self, X_raw: np.ndarray,
                    epsilon: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step FGSM attack.

        Args:
            X_raw: Raw (unscaled) attack samples, shape (N, D)
            epsilon: Perturbation budget (uses self.epsilon if None)

        Returns:
            X_adv_raw: Adversarial examples in original feature space
            perturbations: L∞ perturbation magnitudes per sample
        """
        eps = epsilon if epsilon is not None else self.epsilon

        X_s = self.scaler.transform(X_raw.astype(np.float32))
        X_t = torch.tensor(X_s, dtype=torch.float32, requires_grad=True)
        y_t = torch.ones(len(X_raw), dtype=torch.float32)  # true label = attack

        # Forward pass
        out = self.model(X_t).squeeze()
        loss = self.criterion(out, y_t)

        # Backward pass to get gradients w.r.t. input
        self.model.zero_grad()
        loss.backward()

        # FGSM perturbation: negate sign to minimise attack probability
        # (we want loss to increase for y=0, i.e., fool model → normal)
        grad = self._mask_gradient(X_t.grad.data)
        perturbation = eps * grad.sign()

        # Adversarial example in scaled space
        X_adv_s = (X_t + perturbation).detach().numpy()

        # Clip to valid feature range (prevents nonsensical values)
        X_adv_s = np.clip(X_adv_s, -5.0, 5.0)  # ±5σ in standardised space

        # Inverse transform to original space
        X_adv_raw = self.scaler.inverse_transform(X_adv_s)

        # Clip non-negative features
        non_neg_idx = [i for i, f in enumerate(NUMERIC_FEATURES)
                       if f not in ['serror_rate', 'rerror_rate', 'same_srv_rate',
                                    'diff_srv_rate', 'srv_diff_host_rate']]
        X_adv_raw[:, non_neg_idx] = np.maximum(X_adv_raw[:, non_neg_idx], 0)

        perturbation_magnitudes = np.abs(X_adv_s - X_s).max(axis=1)
        return X_adv_raw, perturbation_magnitudes

    def attack_pgd(self, X_raw: np.ndarray,
                   epsilon: Optional[float] = None,
                   n_steps: int = 10,
                   step_size: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projected Gradient Descent (PGD) — iterative FGSM.
        Much stronger attack: multiple small FGSM steps.

        Args:
            X_raw: Raw attack samples
            epsilon: Total perturbation budget
            n_steps: Number of PGD iterations
            step_size: Step size per iteration (default: epsilon/n_steps * 2)

        Returns:
            X_adv_raw: Adversarial examples
            perturbations: L∞ perturbation per sample
        """
        eps = epsilon if epsilon is not None else self.epsilon
        alpha = step_size if step_size is not None else eps / n_steps * 2

        X_s_orig = self.scaler.transform(X_raw.astype(np.float32))
        X_adv_s = X_s_orig.copy()

        for step in range(n_steps):
            X_t = torch.tensor(X_adv_s, dtype=torch.float32, requires_grad=True)
            y_t = torch.ones(len(X_raw), dtype=torch.float32)

            out = self.model(X_t).squeeze()
            loss = self.criterion(out, y_t)
            self.model.zero_grad()
            loss.backward()

            grad = self._mask_gradient(X_t.grad.data).numpy()
            X_adv_s = X_adv_s + alpha * np.sign(grad)

            # Project back into ε-ball
            delta = np.clip(X_adv_s - X_s_orig, -eps, eps)
            X_adv_s = np.clip(X_s_orig + delta, -5.0, 5.0)

        X_adv_raw = self.scaler.inverse_transform(X_adv_s)
        non_neg_idx = list(range(len(NUMERIC_FEATURES)))
        X_adv_raw[:, non_neg_idx] = np.maximum(X_adv_raw[:, non_neg_idx], 0)

        perturbation_magnitudes = np.abs(X_adv_s - X_s_orig).max(axis=1)
        return X_adv_raw, perturbation_magnitudes

    def evaluate_evasion(self, X_raw: np.ndarray,
                         X_adv_raw: np.ndarray,
                         epsilon: float,
                         attack_type: str = "FGSM",
                         n_steps: int = 1) -> AttackResult:
        """
        Measure evasion effectiveness.

        An attack sample evades detection if:
          - Originally classified as Attack (1)
          - After perturbation classified as Normal (0)
        """
        # Original predictions (should all be 1 = attack)
        X_s = self.scaler.transform(X_raw.astype(np.float32))
        X_adv_s = self.scaler.transform(X_adv_raw.astype(np.float32))

        with torch.no_grad():
            orig_proba = self.model(torch.tensor(X_s)).squeeze().numpy()
            adv_proba = self.model(torch.tensor(X_adv_s)).squeeze().numpy()

        orig_preds = (orig_proba >= 0.5).astype(int)
        adv_preds = (adv_proba >= 0.5).astype(int)

        # Only count samples correctly classified as attacks originally
        originally_detected = orig_preds == 1
        evaded = (orig_preds == 1) & (adv_preds == 0)

        n_original = originally_detected.sum()
        n_evaded = evaded.sum()
        evasion_rate = n_evaded / n_original if n_original > 0 else 0.0

        perturbations = np.abs(X_adv_s - X_s)
        avg_l2 = np.linalg.norm(perturbations, axis=1).mean()
        avg_linf = perturbations.max(axis=1).mean()
        conf_drop = (orig_proba[originally_detected] -
                     adv_proba[originally_detected]).mean()

        return AttackResult(
            epsilon=epsilon,
            n_original=int(n_original),
            n_evaded=int(n_evaded),
            evasion_rate=float(evasion_rate),
            avg_perturbation_l2=float(avg_l2),
            avg_perturbation_linf=float(avg_linf),
            confidence_drop=float(conf_drop),
            iterations=n_steps,
            attack_type=attack_type
        )


# ─────────────────────────────────────────────
#  EPSILON SWEEP: Run attack at multiple budgets
# ─────────────────────────────────────────────
def epsilon_sweep(attacker: FGSMAttacker,
                  X_attacks: np.ndarray,
                  epsilons: List[float],
                  attack_type: str = "FGSM",
                  save_dir: str = '../results') -> List[AttackResult]:
    """
    Run FGSM/PGD attacks across a range of ε values.
    Returns list of AttackResult objects.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []

    print(f"\n{'='*60}")
    print(f"  ε-Sweep: {attack_type} Attack on {len(X_attacks)} samples")
    print(f"{'='*60}")
    print(f"  {'ε':>8} | {'Evaded':>8} | {'Evasion%':>10} | "
          f"{'L∞ Pert':>9} | {'Conf Drop':>10}")
    print(f"  {'-'*55}")

    for eps in epsilons:
        attacker.epsilon = eps
        if attack_type == "FGSM":
            X_adv, _ = attacker.attack_fgsm(X_attacks, epsilon=eps)
            result = attacker.evaluate_evasion(X_attacks, X_adv, eps, "FGSM", 1)
        else:  # PGD
            X_adv, _ = attacker.attack_pgd(X_attacks, epsilon=eps, n_steps=10)
            result = attacker.evaluate_evasion(X_attacks, X_adv, eps, "PGD", 10)

        results.append(result)
        print(f"  {eps:>8.3f} | {result.n_evaded:>8d} | "
              f"{result.evasion_rate*100:>9.1f}% | "
              f"{result.avg_perturbation_linf:>9.4f} | "
              f"{result.confidence_drop:>10.4f}")

    # Save results
    results_data = [r.to_dict() for r in results]
    tag = attack_type.lower()
    with open(f'{save_dir}/attack_results_{tag}.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n  Results saved to {save_dir}/attack_results_{tag}.json")
    return results
