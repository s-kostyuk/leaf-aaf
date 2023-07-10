#!/usr/bin/env python3
import itertools
import argparse
import pprint

from train_common import train_variant
from misc import NetInfo


def main():
    parser = argparse.ArgumentParser(
        prog='train_new_simple'
    )
    parser.add_argument('af_type')
    parser.add_argument('--opt', default='rmsprop')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--p24sl', action='store_true')
    parser.add_argument('--dspu4', action='store_true')
    parser.add_argument('--dev', default='gpu')
    parser.add_argument('--net', default='all')
    parser.add_argument('--ds', default='all')
    parser.add_argument('--start_ep', default=0, type=int)
    parser.add_argument('--end_ep', default=100, type=int)
    parser.add_argument('--patch_base', action='store_true')
    parser.add_argument('--acts', default='all_lus')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    net_names = ["LeNet-5", "KerasNet"] if args.net == 'all' else [args.net]
    ds_names = ["F-MNIST", "CIFAR-10"] if args.ds == 'all' else [args.ds]

    if args.acts == 'all':
        act_names = ['ReLU', 'SiLU', 'Tanh', 'Sigmoid']
    elif args.acts == 'all_lus':
        act_names = ['ReLU', 'SiLU']
    elif args.acts == 'all_bfs':
        act_names = ['Tanh', 'Sigmoid']
    else:
        act_names = [args.acts]

    net_ds_combinations = itertools.product(net_names, ds_names, act_names)
    start_ep = args.start_ep
    end_ep = args.end_ep
    nets = []
    opt = args.opt
    seed = args.seed
    bs = args.bs
    p24sl = args.p24sl
    dspu4 = args.dspu4
    dev_name = args.dev
    patch_base = args.patch_base
    wandb = args.wandb

    for n, ds, act in net_ds_combinations:
        net = NetInfo(n, args.af_type, ds, act, end_ep, dspu4=dspu4, opt_name=opt, p24sl=p24sl)
        nets.append(net)

    print("Training the following combinations:")
    pprint.pprint(nets)

    for net in nets:
        train_variant(
            net.net_name, net.net_type, net.ds_name, af_name=net.af_name,
            end_epoch=net.epoch, dspu4=net.dspu4, p24sl=net.p24sl,
            opt_name=net.opt_name, seed=seed, bs=bs, dev_name=dev_name,
            start_epoch=start_ep, patch_base=patch_base, wandb_enable=wandb
        )


if __name__ == "__main__":
    main()
