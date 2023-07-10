import torch


def silu_manual(x):
    return x * torch.sigmoid(x)
