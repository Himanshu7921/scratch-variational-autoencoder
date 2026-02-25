import torch
import torch.nn as nn
import math

class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.tensor):
        return x @ self.weight + self.bias