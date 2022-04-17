import torch.nn as nn
import torch


class MultiLayerQ(nn.Module):
    """Q network consisting of an MLP."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1),nn.Tanh())

    def forward(self, features, actions):
        cat_input = torch.cat((features, actions),1)
        return self.net(cat_input)
