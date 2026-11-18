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
        self.fc1 = nn.Linear(2, 16) # 2 inputs, -> 16 outputs
        self.fc2 = nn.Linear(16, 8) # 16 inputs -> 8 outputs
        self.fc3 = nn.Linear(8, 1)  # 8 inputs -> 1 output
        self.relu = nn.ReLU()

        #raise NotImplementedError("implement MoonsNet.__init__")

    def forward(self, x):
        # TODO: return self.fc3(relu(self.fc2(relu(self.fc1(x)))))
        # ReLU replace each negative value with 0

        x = self.fc1(x) # through layer 1
        x = self.relu(x)
        x = self.fc2(x) # through layer 2
        x = self.relu(x)
        x = self.fc3(x) # through layer 3

        return x

        #raise NotImplementedError("implement MoonsNet.forward"