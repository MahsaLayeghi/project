"""Deterministic moons dataset — do not modify."""
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def make_dataset(n_samples: int = 300, noise: float = 0.20, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_np = StandardScaler().fit_transform(X_np)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y
