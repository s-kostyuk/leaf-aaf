#!/usr/bin/env python3
import itertools

from train_common import train_variant
from misc import NetInfo


def main():
    net_names = ["LeNet-5", "KerasNet"]
    ds_names = ["F-MNIST", "CIFAR-10"]
    net_ds_combinations = itertools.product(net_names, ds_names)
    epochs = 100
    nets = []

    for n, ds in net_ds_combinations:
        nets_nds = [
            NetInfo(n, "base", ds, "Tanh", epochs, dspu4=False),
            NetInfo(n, "leaf", ds, "Tanh", epochs, dspu4=False),
            NetInfo(n, "leaf", ds, "Tanh", epochs, dspu4=True),
        ]
        nets.extend(nets_nds)

    for net in nets:
        train_variant(
            net.net_name, net.net_type, net.ds_name, af_name=net.af_name,
            end_epoch=net.epoch, dspu4=net.dspu4
        )


if __name__ == "__main__":
    main()
