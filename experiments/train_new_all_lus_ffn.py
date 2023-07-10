#!/usr/bin/env python3
import itertools

from train_common import train_variant
from misc import NetInfo


def main():
    net_names = ["LeNet-5", "KerasNet"]
    ds_names = ["F-MNIST", "CIFAR-10"]
    af_names = ["ReLU", "SiLU"]
    combinations = itertools.product(net_names, ds_names, af_names)
    epochs = 100
    nets = []

    for n, ds, af in combinations:
        nets_nds = [
            NetInfo(n, "leaf_ffn", ds, af, epochs, dspu4=False),
        ]
        nets.extend(nets_nds)

    for net in nets:
        train_variant(
            net.net_name, net.net_type, net.ds_name, af_name=net.af_name,
            end_epoch=net.epoch, dspu4=net.dspu4
        )


if __name__ == "__main__":
    main()
