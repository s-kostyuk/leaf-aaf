import torch


def tanh_manual(x):
    return (
        (torch.exp(x) - torch.exp(-x))
        /
        (torch.exp(x) + torch.exp(-x))
    )
