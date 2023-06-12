import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Initialize a residual block with two convolutions
    """
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, (3, 3), padding=1)

    def convblock(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x): return x + self.convblock(x)  # skip connection


class CongCNN(nn.Module):
    def __init__(self, param):
        super().__init__()
        
