# Adversarial Input Attack on ML-based IDS
## CDAC ITISS Project Report

**Team Project** | Centre for Development of Advanced Computing (CDAC)  
**Programme:** Information Technology for Intelligent Security Systems (ITISS)  

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction & Motivation](#introduction)
3. [Background Theory](#theory)
4. [System Architecture](#architecture)
5. [Dataset & Features](#dataset)
6. [IDS Model (IDSNet)](#ids-model)
7. [Adversarial Attack: FGSM & PGD](#attack)
8. [Defence: Adversarial Training](#defence)
9. [Experimental Results](#results)
10. [Why Clean Accuracy ≠ Robustness](#clean-vs-robust)
11. [What Adversarial Training Actually Provides](#adv-training-analysis)
12. [Conclusions](#conclusions)
13. [References](#references)

---

## 1. Abstract <a name="abstract"></a>

Machine-learning based Intrusion Detection Systems (IDS) promise superior detection over signature-based approaches by generalising to unseen attacks. However, this paper demonstrates a critical vulnerability: ML-based IDS models that achieve 100% accuracy on clean network flow data can be **systematically evaded** using adversarial perturbations. We implement:

- A **flow-based neural IDS** (IDSNet) trained on 38 NSL-KDD-style features
- **FGSM** (Fast Gradient Sign Method) and **PGD** (Projected Gradient Descent) attacks that perturb network flow features to evade detection
- **Adversarial training** as a mitigation, demonstrating up to **50.6 percentage point** reduction in evasion rate at ε=1.5

Our findings confirm that clean accuracy is an unreliable proxy for security robustness, and that adversarial training provides meaningful but imperfect defence.

---

## 2. Introduction & Motivation <a name="introduction"></a>

### Why ML-based IDS?

Traditional signature-based IDS (like Snort, Suricata) detect known attacks by matching packet patterns to a database of signatures. Their limitations:

- **Zero-day blindness:** Cannot detect attacks with no existing signature
- **Evasion:** Simple obfuscation (changing byte patterns, encoding) defeats signatures
- **Maintenance:** Requires constant manual signature updates

ML-based IDS overcome these by **learning decision boundaries** from historical traffic patterns. A neural network trained on flow-level statistics (bytes per second, connection counts, error rates) can theoretically generalise to novel attack patterns.

### The Hidden Vulnerability

But ML models are vulnerable to a class of attacks invisible to human analysts: **adversarial inputs**. By making imperceptibly small changes to input features, an attacker can flip the model's decision from "Attack" to "Normal" — while the underlying malicious behaviour continues undetected.

```
                    ┌─────────────────────────────────┐
 Attack traffic ────►  ML IDS Model (sees 38 features) ├──► "ATTACK" ✗ DETECTED
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
 Attack traffic     │                                 │
 + FGSM noise  ────►  ML IDS Model (same 38 features) ├──► "NORMAL" ✓ EVADED
                    └─────────────────────────────────┘
```

The adversary modifies observable flow-level statistics (connection count, error rate, byte counts) slightly — staying within plausible traffic bounds — but the perturbation is precisely calculated to maximise the model's confusion.

---

## 3. Background Theory <a name="theory"></a>

### 3.1 Adversarial Examples (Goodfellow et al., 2014)

An adversarial example is an input `x_adv` derived from clean input `x` such that:
- `||x_adv - x||_p ≤ ε` (small change, bounded by epsilon)
- `f(x_adv) ≠ f(x)` (model prediction changes)

For neural networks, the gradient ∇_x J(θ, x, y) tells us **which direction in feature space increases the loss**. Moving in that direction pushes the input towards the wrong class.

### 3.2 FGSM (Fast Gradient Sign Method)

```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

- `J`: Cross-entropy loss (BCE for binary classification)
- `ε`: Perturbation budget (controls how much we can change features)
- `sign(·)`: Takes only the direction, ensuring uniform L∞ bound

**In our IDS context:**
- `x` = scaled network flow feature vector (38 dimensions)
- `y` = attack label (1)
- Goal: perturb x so that `J(θ, x_adv, y=1)` is maximised
- This causes the model to misclassify the attack as normal traffic

### 3.3 PGD (Projected Gradient Descent)

FGSM is a single-step attack. PGD applies multiple small FGSM steps, projecting back onto the ε-ball after each:

```
x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x J(θ, x^(t), y)))
```

- `α` = step size (typically ε/T × 2)
- `Π` = projection operator (clips to ε-ball)
- PGD is considered the **strongest first-order attack**

### 3.4 Adversarial Training (Madry et al., 2018)

The defence is to train on adversarial examples:

```
min_θ E_{(x,y)~D} [ max_{δ: ||δ||≤ε} J(θ, x+δ, y) ]
```

This is a **min-max problem**: the inner maximisation finds the worst-case perturbation; the outer minimisation trains the model to be robust to it.

**Practical implementation:** At each training batch, generate FGSM-perturbed versions and include them in training, mixing clean and adversarial examples 50/50.

### 3.5 Threat Model

| Property | Value |
|----------|-------|
| Adversary knowledge | **White-box** (knows model weights and architecture) |
| Adversary goal | Evasion — bypass detection while maintaining attack functionality |
| Constraint | Only **mutable flow features** may be perturbed (22/38 features) |
| Perturbation norm | L∞ (bounded maximum feature change) |

Immutable features (IP flags like `land`, kernel-level fields) are excluded — an attacker cannot easily modify them without breaking packet validity.

---

## 4. System Architecture <a name="architecture"></a>

```
┌───────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                       │
│                                                           │
│  NSL-KDD Data → StandardScaler → IDSNet → BCE Loss       │
│                                    ↑                      │
│                          (Standard Training)              │
│                                                           │
│  NSL-KDD Data → StandardScaler → IDSNet ──┬──→ BCE Loss  │
│                                    ↑      │               │
│                               FGSM Attack │               │
│                              (Adv Training) ←─────────┘  │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                     ATTACK PIPELINE                        │
│                                                           │
│  Attack Samples → StandardScaler → IDSNet                │
│                                       ↓                   │
│                              Compute ∇_x J                │
│                                       ↓                   │
│                       x_adv = x + ε·sign(∇_x J)          │
│                                       ↓                   │
│                    Apply mutable-feature mask              │
│                                       ↓                   │
│                    Clip to valid feature ranges            │
│                                       ↓                   │
│                    IDSNet(x_adv) → "Normal"? EVADED       │
└───────────────────────────────────────────────────────────┘
```

---

## 5. Dataset & Features <a name="dataset"></a>

### NSL-KDD Dataset Structure

The NSL-KDD dataset (improved KDD Cup 1999) contains **41 features** extracted from TCP/IP connections. We use **38 numeric features** after dropping categorical fields encoded separately.

**Feature categories:**

| Category | Features | Examples |
|----------|----------|---------|
| Basic TCP | 4 | `duration`, `src_bytes`, `dst_bytes` |
| Content | 13 | `logged_in`, `num_failed_logins`, `root_shell` |
| Traffic (same host) | 9 | `count`, `serror_rate`, `same_srv_rate` |
| Traffic (destination host) | 10 | `dst_host_count`, `dst_host_serror_rate` |

**Label distribution (our synthetic data):**
```
Total samples : 20,000
Normal traffic: 10,000 (50.0%)
Attack traffic: 10,000 (50.0%)
```

### Mutable vs Immutable Features

Of 38 features, 22 are marked **mutable** (can be manipulated by an attacker):

- **Mutable:** `duration`, `src_bytes`, `dst_bytes`, connection counts, error rates, service rates, destination host statistics
- **Immutable:** `land` (loopback attack flag), `urgent` (TCP urgent pointer), `root_shell`, `su_attempted`, `num_root` (these reflect actual system-level compromise, not observable flow stats)

---

## 6. IDS Model (IDSNet) <a name="ids-model"></a>

### Architecture

```
Input (38) → BatchNorm1d(38) → Linear(256) → ReLU → Dropout(0.3)
           → Linear(128) → ReLU → Dropout(0.3)
           → Linear(64) → ReLU
           → Linear(1) → Sigmoid
```

**Design choices:**
- **BatchNorm1d at input:** Normalises within a batch, reduces sensitivity to feature scale drift
- **Dropout (0.3):** Reduces co-adaptation between neurons, important for robustness
- **3-layer depth:** Deep enough to learn non-linear decision boundaries, shallow enough to remain interpretable
- **Sigmoid output:** Produces attack probability ∈ [0,1]; threshold at 0.5

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss | Binary Cross-Entropy |
| Batch size | 256 |
| Epochs | 25 |
| LR Schedule | StepLR (γ=0.5, step=10) |
| Train/Val split | 80/20 |

### Clean Data Performance

```
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00      2000
      Attack       1.00      1.00      1.00      2000

    accuracy                           1.00      4000
```

**Critical observation:** 100% accuracy on clean data — yet the model is highly vulnerable to adversarial inputs (see Section 9).

---

## 7. Adversarial Attack: FGSM & PGD <a name="attack"></a>

### 7.1 FGSM Implementation

```python
def attack_fgsm(self, X_raw, epsilon):
    # 1. Scale features
    X_s = self.scaler.transform(X_raw)
    X_t = torch.tensor(X_s, requires_grad=True)
    y_t = torch.ones(len(X_raw))  # true label = attack

    # 2. Forward pass + loss
    out = self.model(X_t).squeeze()
    loss = criterion(out, y_t)

    # 3. Backward pass — compute ∇_x J
    loss.backward()

    # 4. Apply FGSM (gradient sign, masked to mutable features only)
    grad = mask_to_mutable_features(X_t.grad)
    perturbation = epsilon * grad.sign()

    # 5. Adversarial example
    X_adv = X_t + perturbation

    return inverse_transform(X_adv)  # back to original space
```

### 7.2 PGD Implementation

```python
def attack_pgd(self, X_raw, epsilon, n_steps=10):
    alpha = epsilon / n_steps * 2  # step size
    X_adv = X_orig.copy()

    for step in range(n_steps):
        # FGSM step
        X_adv = X_adv + alpha * sign(∇_x J)
        # Project back to ε-ball
        delta = clip(X_adv - X_orig, -epsilon, epsilon)
        X_adv = clip(X_orig + delta, -5σ, +5σ)

    return inverse_transform(X_adv)
```

### 7.3 Realistic Constraints Applied

After generating adversarial examples, we apply post-processing to maintain packet validity:
1. **Non-negativity:** Byte counts, connection counts ≥ 0
2. **Rate bounds:** Error rates, service rates ∈ [0, 1]
3. **Integer rounding:** Count fields rounded to integers
4. **Feature clipping:** All values within ±5 standard deviations in scaled space

---

## 8. Defence: Adversarial Training <a name="defence"></a>

### Implementation

During adversarial training, each batch is augmented:

```python
if adversarial_training:
    # Generate FGSM adversarial batch
    X_batch_adv = fgsm_attack(X_batch, y_batch, epsilon=0.15)
    
    # Combine clean + adversarial
    X_combined = concat([X_batch, X_batch_adv])
    y_combined = concat([y_batch, y_batch])
    
    # Train on both
    loss = BCE(model(X_combined), y_combined)
```

### Why This Works

Adversarial training **reshapes the loss landscape** around each training point. The model learns that the correct label should hold not just at `x`, but also within an ε-neighbourhood of `x`. This forces the model to develop **smoother, more conservative decision boundaries**.

```
Standard training:              Adversarial training:
                                
  Normal  |  Attack              Normal  |  Attack
    ──────┼──────                ──────←─┼─→──────
    sharp │                      buffer  │  buffer
    boundary                     zone    │  zone
```

The buffer zone means the attacker needs a much larger perturbation to cross the decision boundary.

---

## 9. Experimental Results <a name="results"></a>

### 9.1 Evasion Rate vs Perturbation Budget (ε)

| ε | Std-FGSM | Adv-FGSM | Std-PGD | Adv-PGD | Defence Gain (FGSM) |
|---|----------|----------|---------|---------|---------------------|
| 0.1 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0 pp |
| 0.3 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0 pp |
| 0.5 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0 pp |
| 0.8 | 0.6% | 1.0% | 0.6% | 2.2% | −0.4 pp |
| 1.0 | 8.4% | 11.0% | 8.4% | 16.0% | −2.6 pp |
| 1.5 | **87.4%** | **36.8%** | **86.8%** | **41.8%** | **+50.6 pp** |
| 2.0 | 95.8% | 55.4% | 95.8% | 61.4% | +40.4 pp |

### 9.2 Key Observations

**1. Phase transition:** Both models resist attack at low ε, but exhibit a sharp jump around ε=1.0–1.5. This reflects the feature distribution gap between normal and attack classes — a threshold that must be crossed for evasion.

**2. Adversarial training effective at large ε:** At ε=1.5, adversarial training reduces FGSM evasion from 87.4% → 36.8% (50.6 pp reduction). This is substantial.

**3. PGD more aggressive on adversarial model:** At ε=1.0, PGD achieves 16.0% vs FGSM's 11.0% on the adversarially-hardened model — PGD is a stronger threat to defended models.

**4. Both models resist small perturbations:** At ε ≤ 0.5, evasion is near 0%. This suggests the feature distributions are well-separated enough that small perturbations do not cross the decision boundary.

**5. Adversarial training does not eliminate vulnerability:** At ε=2.0, even the hardened model shows 55.4% evasion — **complete robustness is not achieved**. This is consistent with theoretical limitations.

### 9.3 Model Accuracy (Clean Data)

| Model | Clean Accuracy |
|-------|---------------|
| Standard IDS | 100.00% |
| Adversarial IDS | 100.00% |

Both models achieve identical clean accuracy — confirming that **accuracy alone cannot distinguish a robust model from a fragile one**.

---

## 10. Why Clean Accuracy ≠ Robustness <a name="clean-vs-robust"></a>

### The Fundamental Problem

A neural network learns a decision function f: X → {0,1}. Clean accuracy measures:

```
Acc = P[f(x) = y] over the test distribution D
```

But adversarial robustness requires:

```
Rob = P[f(x+δ) = y  ∀ δ with ||δ||≤ε] over D
```

These are **completely different properties**. A model can achieve:
- High Acc + Low Rob: Sharp decision boundaries, fits training distribution perfectly but fragile to perturbation
- Lower Acc + Higher Rob: Smoother boundaries, some clean samples misclassified but much more stable

### Geometric Intuition

In high-dimensional feature space (38D in our case), the "volume" near the decision boundary is enormous. Clean test samples may all be far from the boundary (high accuracy), but adversarial examples exploit the geometry to find nearby boundary crossings.

```
         High-dimensional feature space (38D)
         
         ●●●●●● Normal ●●●●●●
         ●●●●●●●●●●●●●●●●●●●     ← Most clean samples here
                ────────────────  ← Decision boundary
         ████████████████████
         ████ Attack █████████     ← Attack samples
         ████████ x  ████████
                  ↑
                  Adversarial perturbation finds 
                  path to boundary (invisible to humans)
```

### Why ML Practitioners Are Misled

Model A: 99% clean accuracy, evaded by ε=0.01 FGSM  
Model B: 95% clean accuracy, resistant to ε=0.5 FGSM  

A security practitioner would deploy Model A. This is the wrong choice.

**The IDS evaluation standard must include adversarial robustness testing**, not just clean-data metrics.

---

## 11. What Adversarial Training Actually Provides <a name="adv-training-analysis"></a>

### Certified vs Empirical Robustness

Adversarial training provides **empirical robustness**: the model performs better against known attack methods used during training. It does NOT provide:

- **Certified robustness:** No mathematical guarantee that no adversarial example exists within the ε-ball
- **Robustness to unseen attacks:** A new attack method (e.g., TRADES, AutoAttack) may still succeed
- **Transferability protection:** Black-box attacks using surrogate models may bypass it

### The Accuracy-Robustness Trade-off

Adversarial training introduces a **fundamental tension**:

```
Clean Accuracy ↔ Adversarial Robustness
```

Training on perturbed examples forces the model to maintain correct predictions under worst-case inputs. This requires **smoother decision boundaries**, which can sacrifice some margin on clean data. In our experiment, both clean accuracies remained 100% — typical with well-separated data. On real, noisier datasets, expect a 1-5% clean accuracy drop.

### What Adversarial Training Demonstrates

1. **Defence by knowing the attack:** Effective primarily when training uses the same attack type (FGSM training → FGSM defence). PGD attacks partially bypass FGSM-trained models.

2. **Partial security boundary expansion:** Pushes the decision boundary back from training points, creating a robustness margin. But the margin is bounded by the training epsilon.

3. **Arms race limitation:** A sophisticated attacker can increase ε beyond the training budget. Adversarial training at ε_train provides robustness roughly within [0, ε_train]; beyond that, vulnerability remains.

4. **Practical security improvement:** Despite limitations, in our experiment adversarial training reduced attack evasion by 40-50 percentage points at moderate ε. This is significant in real deployments where attackers cannot easily apply very large perturbations without breaking functionality.

### Mitigation Limitations Summary

| Defence | Provides | Does Not Provide |
|---------|---------|-----------------|
| Adversarial Training (FGSM) | Resistance to FGSM attacks up to ε_train | Complete robustness, certified guarantees |
| Adversarial Training (PGD) | Stronger robustness, more general | Resistance to adaptive/white-box attacks |
| Feature Discretisation | Reduces gradient signal | Obfuscated gradient still exploitable |
| Ensemble Methods | Harder to transfer attacks | No single robust solution |
| Certified Defences (IBP, etc.) | Provable bounds | Scale to large networks/features |

---

## 12. Conclusions <a name="conclusions"></a>

This project demonstrates that:

1. **ML-based IDS achieves high clean accuracy but is not inherently robust.** Our IDSNet achieved 100% accuracy on clean network flows yet was evaded up to 95.8% of the time under strong perturbations.

2. **FGSM and PGD attacks effectively evade flow-based IDS** by exploiting gradient information to craft feature-level perturbations. The attack operates in the scaled feature space and respects real-world constraints by masking immutable features.

3. **Adversarial training is a meaningful but incomplete defence.** At ε=1.5, it reduces FGSM evasion from 87.4% to 36.8% — a 50.6 percentage point improvement. However, sufficiently large perturbations (ε=2.0) still achieve 55.4% evasion.

4. **Clean accuracy cannot predict adversarial robustness.** Both standard and adversarially-trained models achieve 100% clean accuracy, yet differ drastically in adversarial performance. Security evaluations must include adversarial robustness benchmarks.

5. **Adversarial ML represents a genuine security threat.** In real deployments, adversaries motivated to evade IDS detection have reason to invest in FGSM-style feature perturbation. Adversarial training should be a standard component of ML-based security system training pipelines.

### Future Work

- Implement **TRADES** and **AutoAttack** for stronger evaluation
- Test on **actual NSL-KDD / CICIDS datasets** with real traffic imbalance
- Explore **certified robustness** methods (randomised smoothing)
- Study **black-box transfer attacks** (no model access)
- Apply to **multiclass IDS** (DoS / Probe / R2L / U2R attack classification)

---

## 13. References <a name="references"></a>

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *ICLR 2015*.

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.

3. Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z., & Swami, A. (2016). The limitations of deep learning in adversarial settings. *IEEE EuroS&P 2016*.

4. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE CISDA 2009*. (NSL-KDD dataset)

5. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. *IEEE S&P 2017*.

6. Zhang, H., Yu, Y., Jiao, J., et al. (2019). Theoretically principled trade-off between robustness and accuracy. *ICML 2019*.

7. Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP 2018*. (CICIDS dataset)

---

*CDAC ITISS | Adversarial Input Attack on ML-based IDS*  
*All code available in `src/` directory. Run `pip install -r requirements.txt` then `python src/run_experiment.py`*
