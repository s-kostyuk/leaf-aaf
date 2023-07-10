from typing import Tuple, Any, Sequence

import torch
from torch.autograd.function import FunctionCtx


class _AHAF(torch.autograd.Function):
    @staticmethod
    def forward(u, beta, gamma) -> Any:
        y = (beta * u) * torch.sigmoid(gamma * u)
        return y

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Sequence[Any], output: Any) -> None:
        u, beta, gamma = inputs
        ctx.save_for_backward(u, beta, gamma)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Any) -> Any:
        u, beta, gamma = ctx.saved_tensors
        grad_u = grad_beta = grad_gamma = None

        gamma_u = gamma * u
        sig_gamma_u = torch.sigmoid(gamma_u)

        if ctx.needs_input_grad[0]:
            grad_u = grad_output.mul(
                (beta * sig_gamma_u)
                *
                (1 + u * gamma * (1 - sig_gamma_u))
            )
        if ctx.needs_input_grad[1]:
            grad_beta = grad_output.mul(u * sig_gamma_u)
        if ctx.needs_input_grad[2]:
            grad_gamma = grad_output.mul(
                (beta * u)
                * sig_gamma_u
                * torch.sigmoid(-gamma_u)
                * u
            )

        return grad_u, grad_beta, grad_gamma


def _ahaf(u, beta, gamma):
    return _AHAF.apply(u, beta, gamma)


class AHAF(torch.nn.Module):
    def __init__(self, *, size: Tuple[int, ...] = (1,), init_as: str = 'ReLU'):
        super(AHAF, self).__init__()

        if init_as == 'ReLU':
            self.gamma = torch.nn.Parameter(torch.ones(*size) * (2.0**16))
            self.beta = torch.nn.Parameter(torch.ones(*size))
        elif init_as == 'SiLU':
            self.gamma = torch.nn.Parameter(torch.ones(*size))
            self.beta = torch.nn.Parameter(torch.ones(*size))
        elif init_as == 'CUSTOM':
            self.gamma = torch.nn.Parameter(torch.ones(*size)*10)
            self.beta = torch.nn.Parameter(torch.ones(*size))
        else:
            raise ValueError("Invalid initialization mode [{}]".format(init_as))

    @staticmethod
    def _get_sample_value(t: torch.Tensor) -> float:
        size = t.size()

        for _ in size:
            t = t[0]

        return t.item()

    def forward(self, inputs):
        return _ahaf(inputs, self.beta, self.gamma)

    def __repr__(self):
        return "AHAF(size={},gamma={}, beta={})".format(
            tuple(self.gamma.size()),
            self._get_sample_value(self.gamma),
            self._get_sample_value(self.beta)
        )
