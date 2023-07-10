from typing import Tuple, Any, Sequence

import torch
from torch.autograd.function import FunctionCtx


class _LEAF(torch.autograd.Function):
    @staticmethod
    def forward(u, p1, p2, p3, p4) -> Any:
        y = (p1 * u + p2) * torch.sigmoid(p3 * u) + p4
        return y

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Sequence[Any], output: Any) -> None:
        u, p1, p2, p3, p4 = inputs
        ctx.save_for_backward(u, p1, p2, p3, p4)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Any) -> Any:
        u, p1, p2, p3, p4 = ctx.saved_tensors
        grad_u = grad_p1 = grad_p2 = grad_p3 = grad_p4 = None

        p3_u = p3 * u
        sig_p3_u = torch.sigmoid(p3_u)

        if ctx.needs_input_grad[0]:
            grad_u = grad_output.mul(
                (p1 * sig_p3_u)
                +
                (p1 * u + p2)
                * sig_p3_u
                * torch.sigmoid(-p3_u)
                * p3
            )
        if ctx.needs_input_grad[1]:
            grad_p1 = grad_output.mul(u * sig_p3_u)
        if ctx.needs_input_grad[2]:
            grad_p2 = grad_output.mul(sig_p3_u)
        if ctx.needs_input_grad[3]:
            grad_p3 = grad_output.mul(
                (p1 * u + p2)
                * sig_p3_u
                * torch.sigmoid(-p3_u)
                * u
            )
        if ctx.needs_input_grad[4]:
            grad_p4 = grad_output.mul(torch.ones_like(u))

        return grad_u, grad_p1, grad_p2, grad_p3, grad_p4


def _leaf(u, p1, p2, p3, p4):
    return _LEAF.apply(u, p1, p2, p3, p4)


class LEAF(torch.nn.Module):
    def __init__(self, *, size: Tuple[int, ...] = (1,), init_as: str = 'ReLU'):
        super(LEAF, self).__init__()

        if init_as == 'ReLU':
            self.p1 = torch.nn.Parameter(torch.ones(*size))
            self.p2 = torch.nn.Parameter(torch.zeros(*size))
            self.p3 = torch.nn.Parameter(torch.ones(*size) * (2.0**16))
            self.p4 = torch.nn.Parameter(torch.zeros(*size))
        elif init_as == 'SiLU':
            self.p1 = torch.nn.Parameter(torch.ones(*size))
            self.p2 = torch.nn.Parameter(torch.zeros(*size))
            self.p3 = torch.nn.Parameter(torch.ones(*size))
            self.p4 = torch.nn.Parameter(torch.zeros(*size))
        elif init_as == 'CUSTOM':
            self.p1 = torch.nn.Parameter(torch.ones(*size))
            self.p2 = torch.nn.Parameter(torch.zeros(*size))
            self.p3 = torch.nn.Parameter(torch.ones(*size) * 10)
            self.p4 = torch.nn.Parameter(torch.zeros(*size))
        elif init_as == 'Tanh':
            self.p1 = torch.nn.Parameter(torch.zeros(*size))
            self.p2 = torch.nn.Parameter(torch.ones(*size) * 2.0)
            self.p3 = torch.nn.Parameter(torch.ones(*size) * 2.0)
            self.p4 = torch.nn.Parameter(torch.ones(*size) * -1.0)
        elif init_as == 'Sigmoid':
            self.p1 = torch.nn.Parameter(torch.zeros(*size))
            self.p2 = torch.nn.Parameter(torch.ones(*size))
            self.p3 = torch.nn.Parameter(torch.ones(*size))
            self.p4 = torch.nn.Parameter(torch.zeros(*size))
        else:
            raise ValueError("Invalid initialization mode [{}]".format(init_as))

    @staticmethod
    def _get_sample_value(t: torch.Tensor) -> float:
        size = t.size()

        for _ in size:
            t = t[0]

        return t.item()

    def forward(self, x):
        y = _leaf(x, self.p1, self.p2, self.p3, self.p4)
        return y

    def __repr__(self):
        return "LEAF(size={},p1={},p2={},p3={},p4={})".format(
            tuple(self.p3.size()),
            self._get_sample_value(self.p1),
            self._get_sample_value(self.p2),
            self._get_sample_value(self.p3),
            self._get_sample_value(self.p4)
        )
