from typing import Callable, Optional

import torch
import torch.nn
import torch.nn.functional

from torch import Tensor

from ..trad import silu_manual


AfTraditional = Callable[[Tensor], Tensor]


def af_build_traditional(
        af_name: str,
        min_val: Optional[float] = None, max_val: Optional[float] = None
) -> AfTraditional:
    if af_name == "ReLU":
        return torch.relu
    elif af_name == "SiLU":
        # Using a custom SiLU implementation to exactly follow AAF alternatives
        return silu_manual
    elif af_name == "Tanh":
        return torch.tanh
    elif af_name == "HardTanh":
        if min_val is None or max_val is None:
            return torch.nn.Hardtanh()
        else:
            return torch.nn.Hardtanh(min_val, max_val)
    elif af_name == "Sigmoid":
        return torch.sigmoid
    elif af_name == "HardSigmoid":
        return torch.nn.functional.hardsigmoid
    else:
        raise NotImplementedError(
            "The requested traditional activation function is not supported"
        )
