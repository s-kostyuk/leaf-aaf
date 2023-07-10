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
    opt = "rmsprop"
    #opt = "adam"

    for n, ds in net_ds_combinations:
        nets_nds = [
            NetInfo(n, "base", ds, "ReLU", epochs, dspu4=False, opt_name=opt),
            NetInfo(n, "base", ds, "SiLU", epochs, dspu4=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "ReLU", epochs, dspu4=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "SiLU", epochs, dspu4=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "ReLU", epochs, dspu4=True, opt_name=opt),
            NetInfo(n, "ahaf", ds, "SiLU", epochs, dspu4=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "ReLU", epochs, dspu4=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "SiLU", epochs, dspu4=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "ReLU", epochs, dspu4=True, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "SiLU", epochs, dspu4=True, p24sl=True, opt_name=opt),
        ]
        nets.extend(nets_nds)

    for net in nets:
        train_variant(
            net.net_name, net.net_type, net.ds_name, af_name=net.af_name,
            end_epoch=net.epoch, dspu4=net.dspu4, p24sl=net.p24sl,
            opt_name=net.opt_name
        )


if __name__ == "__main__":
    main()
