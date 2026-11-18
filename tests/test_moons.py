import json
import os
import subprocess
import sys

import torch

from model import MoonsNet
from data import make_dataset


def test_model_output_shape():
    m = MoonsNet()
    x = torch.zeros(5, 2)
    out = m(x)
    assert out.shape == (5, 1), f"expected (5,1), got {out.shape}"


def test_model_has_three_linear_layers():
    import torch.nn as nn
    m = MoonsNet()
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) >= 3, f"need >=3 Linear layers, found {len(linears)}"
    assert linears[0].in_features == 2, "first Linear should take 2 features"
    assert linears[-1].out_features == 1, "last Linear should output 1 value"


def test_data_shapes():
    X, y = make_dataset()
    assert X.shape == (300, 2)
    assert y.shape == (300, 1)


def test_training_runs_and_final_loss():
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    assert os.path.exists("results.json"), "train.py did not create results.json"

    with open("results.json") as f:
        data = json.load(f)
    assert "final_loss" in data
    assert "history" in data
    assert len(data["history"]) >= 100, "not enough epochs recorded"
    assert data["final_loss"] < 0.35, (
        f"final_loss = {data['final_loss']:.4f} — must be below 0.35."
    )


def test_loss_decreases_meaningfully():
    with open("results.json") as f:
        data = json.load(f)
    history = data["history"]
    early = sum(history[:10]) / 10
    late = sum(history[-10:]) / 10
    assert late < early, f"loss did not decrease: {early:.4f} -> {late:.4f}"
    assert late < early * 0.5, (
        f"loss barely decreased: early {early:.4f} -> late {late:.4f}"
    )
