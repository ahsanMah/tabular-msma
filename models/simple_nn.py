from turtle import forward
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, input_dims, hidden_dim=64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.seq_modules = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dims),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)

    def forward(self, x):
        x = self.modules(x)
        return x
