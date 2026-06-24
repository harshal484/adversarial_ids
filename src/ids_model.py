"""
ids_model.py
============
Flow-based Intrusion Detection System (IDS) using a Neural Network.
Trained on the NSL-KDD / synthetic network flow dataset.

CDAC ITISS Project: Adversarial Input Attack on ML-based IDS
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import json

# ─────────────────────────────────────────────
#  1. NETWORK FLOW FEATURE COLUMNS (NSL-KDD)
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
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


# ─────────────────────────────────────────────
#  2. DEEP NEURAL NETWORK IDS MODEL
# ─────────────────────────────────────────────
class IDSNet(nn.Module):
    """
    Multi-layer Neural Network for binary classification:
    Normal (0) vs Attack (1)
    
    Architecture:
        Input → BN → Dense(256) → ReLU → Dropout(0.3)
               → Dense(128) → ReLU → Dropout(0.3)
               → Dense(64)  → ReLU
               → Dense(1)   → Sigmoid
    """

    def __init__(self, input_dim: int):
        super(IDSNet, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  3. DATA GENERATION (Synthetic NSL-KDD-like)
# ─────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic network flow data mimicking NSL-KDD distribution.
    Normal traffic: low bytes, low error rates.
    Attack traffic: high bytes, high error rates, anomalous patterns.
    """
    np.random.seed(seed)
    n_normal = n_samples // 2
    n_attack = n_samples - n_normal

    def make_normal(n):
        data = {
            'duration': np.random.exponential(2, n),
            'src_bytes': np.random.exponential(500, n),
            'dst_bytes': np.random.exponential(1000, n),
            'land': np.zeros(n),
            'wrong_fragment': np.random.poisson(0.01, n),
            'urgent': np.zeros(n),
            'hot': np.random.poisson(0.5, n),
            'num_failed_logins': np.zeros(n),
            'logged_in': np.random.binomial(1, 0.7, n),
            'num_compromised': np.zeros(n),
            'root_shell': np.zeros(n),
            'su_attempted': np.zeros(n),
            'num_root': np.zeros(n),
            'num_file_creations': np.random.poisson(0.1, n),
            'num_shells': np.zeros(n),
            'num_access_files': np.random.poisson(0.2, n),
            'num_outbound_cmds': np.zeros(n),
            'is_host_login': np.zeros(n),
            'is_guest_login': np.random.binomial(1, 0.05, n),
            'count': np.random.randint(1, 50, n),
            'srv_count': np.random.randint(1, 50, n),
            'serror_rate': np.random.beta(1, 10, n),
            'srv_serror_rate': np.random.beta(1, 10, n),
            'rerror_rate': np.random.beta(1, 20, n),
            'srv_rerror_rate': np.random.beta(1, 20, n),
            'same_srv_rate': np.random.beta(8, 2, n),
            'diff_srv_rate': np.random.beta(1, 8, n),
            'srv_diff_host_rate': np.random.beta(1, 5, n),
            'dst_host_count': np.random.randint(1, 255, n),
            'dst_host_srv_count': np.random.randint(1, 255, n),
            'dst_host_same_srv_rate': np.random.beta(7, 3, n),
            'dst_host_diff_srv_rate': np.random.beta(1, 7, n),
            'dst_host_same_src_port_rate': np.random.beta(5, 5, n),
            'dst_host_srv_diff_host_rate': np.random.beta(1, 5, n),
            'dst_host_serror_rate': np.random.beta(1, 15, n),
            'dst_host_srv_serror_rate': np.random.beta(1, 15, n),
            'dst_host_rerror_rate': np.random.beta(1, 20, n),
            'dst_host_srv_rerror_rate': np.random.beta(1, 20, n),
            'label': np.zeros(n, dtype=int)
        }
        return pd.DataFrame(data)

    def make_attack(n):
        data = {
            'duration': np.random.exponential(0.5, n),
            'src_bytes': np.random.exponential(50000, n),
            'dst_bytes': np.random.exponential(100, n),
            'land': np.random.binomial(1, 0.1, n),
            'wrong_fragment': np.random.poisson(1.5, n),
            'urgent': np.random.binomial(1, 0.05, n),
            'hot': np.random.poisson(5, n),
            'num_failed_logins': np.random.poisson(2, n),
            'logged_in': np.random.binomial(1, 0.2, n),
            'num_compromised': np.random.poisson(3, n),
            'root_shell': np.random.binomial(1, 0.3, n),
            'su_attempted': np.random.binomial(1, 0.2, n),
            'num_root': np.random.poisson(2, n),
            'num_file_creations': np.random.poisson(2, n),
            'num_shells': np.random.poisson(1, n),
            'num_access_files': np.random.poisson(3, n),
            'num_outbound_cmds': np.zeros(n),
            'is_host_login': np.random.binomial(1, 0.05, n),
            'is_guest_login': np.random.binomial(1, 0.2, n),
            'count': np.random.randint(200, 512, n),
            'srv_count': np.random.randint(1, 512, n),
            'serror_rate': np.random.beta(8, 2, n),
            'srv_serror_rate': np.random.beta(8, 2, n),
            'rerror_rate': np.random.beta(5, 5, n),
            'srv_rerror_rate': np.random.beta(5, 5, n),
            'same_srv_rate': np.random.beta(2, 8, n),
            'diff_srv_rate': np.random.beta(6, 4, n),
            'srv_diff_host_rate': np.random.beta(5, 5, n),
            'dst_host_count': np.random.randint(1, 10, n),
            'dst_host_srv_count': np.random.randint(1, 10, n),
            'dst_host_same_srv_rate': np.random.beta(2, 8, n),
            'dst_host_diff_srv_rate': np.random.beta(6, 4, n),
            'dst_host_same_src_port_rate': np.random.beta(8, 2, n),
            'dst_host_srv_diff_host_rate': np.random.beta(5, 5, n),
            'dst_host_serror_rate': np.random.beta(7, 3, n),
            'dst_host_srv_serror_rate': np.random.beta(7, 3, n),
            'dst_host_rerror_rate': np.random.beta(5, 5, n),
            'dst_host_srv_rerror_rate': np.random.beta(5, 5, n),
            'label': np.ones(n, dtype=int)
        }
        return pd.DataFrame(data)

    df = pd.concat([make_normal(n_normal), make_attack(n_attack)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  4. IDS TRAINER CLASS
# ─────────────────────────────────────────────
class IDSTrainer:
    def __init__(self, model_dir: str = '../models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.input_dim = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def preprocess(self, df: pd.DataFrame):
        """Scale numeric features and return tensors."""
        X = df[NUMERIC_FEATURES].values.astype(np.float32)
        y = df['label'].values.astype(np.float32)
        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 30, batch_size: int = 256,
              lr: float = 1e-3, adversarial: bool = False, fgsm_epsilon: float = 0.1):
        """
        Train the IDS neural network.
        If adversarial=True, uses adversarial training (FGSM augmentation).
        """
        X, y = self.preprocess(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit scaler on training data only
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        self.input_dim = X_train_s.shape[1]
        self.model = IDSNet(self.input_dim)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_ds = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        print(f"\n{'='*60}")
        print(f"  Training {'Adversarially-Hardened' if adversarial else 'Standard'} IDS Model")
        print(f"  Samples: {len(X_train)} train / {len(X_val)} val")
        print(f"  Architecture: {self.input_dim} → 256 → 128 → 64 → 1")
        print(f"{'='*60}")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                if adversarial:
                    # FGSM adversarial augmentation during training
                    X_batch_adv = self._fgsm_batch(X_batch, y_batch, criterion, fgsm_epsilon)
                    # Mix clean + adversarial
                    X_combined = torch.cat([X_batch, X_batch_adv], dim=0)
                    y_combined = torch.cat([y_batch, y_batch], dim=0)
                    out = self.model(X_combined).squeeze()
                    loss = criterion(out, y_combined)
                else:
                    out = self.model(X_batch).squeeze()
                    loss = criterion(out, y_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_t).squeeze()
                val_loss = criterion(val_out, y_val_t).item()
                val_preds = (val_out >= 0.5).float()
                val_acc = (val_preds == y_val_t).float().mean().item()

            avg_loss = epoch_loss / len(train_loader)
            self.history['train_loss'].append(avg_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1:3d}/{epochs}] "
                      f"Train Loss: {avg_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc*100:.2f}%")

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_out = self.model(X_val_t).squeeze()
            final_preds = (val_out >= 0.5).numpy()

        print(f"\n{'='*60}")
        print("  Final Classification Report:")
        print(classification_report(y_val, final_preds,
                                    target_names=['Normal', 'Attack']))

        # Save
        tag = 'adversarial' if adversarial else 'standard'
        torch.save(self.model.state_dict(),
                   f'{self.model_dir}/ids_{tag}.pth')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler_{tag}.pkl')

        metrics = {
            'accuracy': float(accuracy_score(y_val, final_preds)),
            'type': tag,
            'input_dim': self.input_dim,
            'epochs': epochs
        }
        with open(f'{self.model_dir}/metrics_{tag}.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  Model saved to {self.model_dir}/ids_{tag}.pth")
        return metrics

    def _fgsm_batch(self, X_batch, y_batch, criterion, epsilon):
        """Apply FGSM to a batch for adversarial training."""
        X_adv = X_batch.clone().detach().requires_grad_(True)
        out = self.model(X_adv).squeeze()
        loss = criterion(out, y_batch)
        loss.backward()
        perturbation = epsilon * X_adv.grad.sign()
        return (X_adv + perturbation).detach()

    def load(self, tag: str = 'standard'):
        """Load a saved model and scaler."""
        self.scaler = joblib.load(f'{self.model_dir}/scaler_{tag}.pkl')
        with open(f'{self.model_dir}/metrics_{tag}.json') as f:
            meta = json.load(f)
        self.input_dim = meta['input_dim']
        self.model = IDSNet(self.input_dim)
        self.model.load_state_dict(
            torch.load(f'{self.model_dir}/ids_{tag}.pth', map_location='cpu')
        )
        self.model.eval()
        print(f"  Loaded {tag} model (acc={meta['accuracy']*100:.2f}%)")

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """Predict on raw (unscaled) feature arrays. Returns 0/1."""
        X_s = self.scaler.transform(X_raw.astype(np.float32))
        t = torch.tensor(X_s, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(t).squeeze()
        return (out >= 0.5).numpy().astype(int)

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """Return attack probability scores."""
        X_s = self.scaler.transform(X_raw.astype(np.float32))
        t = torch.tensor(X_s, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(t).squeeze()
        return out.numpy()
