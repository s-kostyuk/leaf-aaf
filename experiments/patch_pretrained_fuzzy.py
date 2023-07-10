#!/usr/bin/env python3

import torch
import torch.nn
import torch.utils.data
import torchinfo

from experiments.common import get_device
from misc import get_file_name_net, create_net


def patch_variant_as_fuzzy(net_name: str, ds_name: str, af_name: str):
    batch_size = 64
    rand_seed = 42
    n_epochs_init = 100

    print("Patching the base {} network with {} activation "
          "for {} to use Fuzzy AF".format(net_name, af_name, ds_name))

    path_base = get_file_name_net(
        net_name, "base", ds_name, af_name, n_epochs_init, patched=False
    )

    dev = get_device()
    torch.manual_seed(rand_seed)
    input_size = (batch_size, 3, 32, 32)

    net = create_net(net_name, "fuzzy_ffn", ds_name, af_name)
    net.to(device=dev)
    torchinfo.summary(net, input_size=input_size, device=dev)

    missing, unexpected = net.load_state_dict(
        torch.load(path_base), strict=False
    )

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    path_ahaf = get_file_name_net(
        net_name, "fuzzy_ffn", ds_name, af_name, n_epochs_init, patched=True
    )

    torch.save(
        net.state_dict(),
        path_ahaf
    )


def main():
    af_names = ("Tanh", "Sigmoid")
    net_name = "KerasNet"
    ds_name = "CIFAR-10"

    for af in af_names:
        patch_variant_as_fuzzy(net_name, ds_name, af)


if __name__ == "__main__":
    main()
