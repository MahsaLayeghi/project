"""Train MoonsNet and save results.json.

Run me with:  bash run.sh
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim

from data import make_dataset
from model import MoonsNet


def train_moons(num_epochs: int = 500, lr: float = 0.05, seed: int = 42):
    """Train MoonsNet on the moons dataset. Return (model, history)."""
    torch.manual_seed(seed)
    X, y = make_dataset()

    # TODO: instantiate model, criterion (BCEWithLogitsLoss), optimiser (Adam, lr=lr)
    model = MoonsNet()
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = lr)

    history = []
    for _ in range(num_epochs):
        # TODO: the five lines of a PyTorch training loop
        # 1. zero the gradients
        optimiser.zero_grad() # clear old gradients
        # 2. forward pass -> preds
        preds = model(X) # forward pass
        # 3. loss = criterion(preds, y)
        loss = criterion(preds, y) #how wrong are we?
        # 4. loss.backward()
        loss.backward() # compute gradients
        # 5. optimiser.step()
        optimiser.step() # update the weights
        # ... then append loss.item() to history.
        history.append(loss.item())
        pass

    return model, history


def main():
    model, history = train_moons()
    if not history:
        raise SystemExit("train_moons() returned an empty history — did you fill in the loop?")
    final_loss = history[-1]
    print(f"Epochs: {len(history)}")
    print(f"Loss at epoch 1:   {history[0]:.4f}")
    print(f"Loss at epoch 100: {history[99]:.4f}")
    print(f"Loss at epoch 500: {history[-1]:.4f}")
    print(f"\nFinal training loss: {final_loss:.4f}   (target: < 0.35)")

    with open("results.json", "w") as f:
        json.dump({"final_loss": final_loss, "history": history}, f)
    print("Saved results.json")


if __name__ == "__main__":
    main()