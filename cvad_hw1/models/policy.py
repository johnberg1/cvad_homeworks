import torch.nn as nn


class MultiLayerPolicy(nn.Module):
    """An MLP based policy network"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,2),nn.Tanh())

    def forward(self, features):
        return self.net(features)
