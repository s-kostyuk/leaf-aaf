#!/usr/bin/env python3

from typing import Iterable, Union
import torch.nn

from nns_aaf import KerasNetAaf, LeNetAaf


def param_freeze_status_to_symbol(param: torch.nn.Parameter) -> str:
    if param.requires_grad:
        return '🟢'
    else:
        return '🔴'


def print_params_freeze_status(params: Iterable[torch.nn.Parameter]):
    status = [
        param_freeze_status_to_symbol(p) for p in params
    ]
    print(status)


def freeze_non_af(net: Union[KerasNetAaf, LeNetAaf]):
    for p in net.parameters():
        p.requires_grad = False

    for afp in net.activation_params:
        afp.requires_grad = True

    print_params_freeze_status(net.parameters())
