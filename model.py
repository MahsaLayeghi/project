"""MoonsNet — fill in the TODOs.

Architecture:
    Input (2) → Linear(2, 16) → ReLU →
    Linear(16, 8) → ReLU →
    Linear(8, 1)   (raw logit — no sigmoid here!)
"""
import torch
import torch.nn as nn


class MoonsNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define self.fc1, self.fc2, self.fc3 with the shapes shown above.
        raise NotImplementedError("implement MoonsNet.__init__")

    def forward(self, x):
        # TODO: return self.fc3(relu(self.fc2(relu(self.fc1(x)))))
        raise NotImplementedError("implement MoonsNet.forward")
