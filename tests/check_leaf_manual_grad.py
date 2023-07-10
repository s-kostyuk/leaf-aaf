import torch
from torch.autograd import gradcheck

from adaptive_afs.cont.leaf import _leaf


def main():
    w = 16
    h = 21

    ins = (
        torch.randn(w, h, dtype=torch.double, requires_grad=True),
        torch.randn(w, h, dtype=torch.double, requires_grad=True),
        torch.randn(w, h, dtype=torch.double, requires_grad=True),
        torch.randn(w, h, dtype=torch.double, requires_grad=True),
        torch.randn(w, h, dtype=torch.double, requires_grad=True),
    )
    test = gradcheck(_leaf, ins, eps=1e-6, atol=1e-4)
    print(test)


if __name__ == "__main__":
    main()
