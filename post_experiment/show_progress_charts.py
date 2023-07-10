#!/usr/bin/env python3

import itertools
import csv
import matplotlib.pyplot as plt

from typing import Sequence, Tuple, List, Union
from cycler import cycler

from misc import ProgressElement
from misc import NetInfo
from misc import get_file_name_stat, get_file_name_stat_img


def load_results(file_path: str) -> List[ProgressElement]:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_legend_long(net_info: Union[Tuple, NetInfo]) -> str:
    if not isinstance(net_info, NetInfo):
        net_info = NetInfo(*net_info)

    af_name_ffn = net_info.af_name

    if net_info.af_name_cnn is None:
        af_name_cnn = af_name_ffn
    else:
        af_name_cnn = net_info.af_name_cnn

    if net_info.net_type == "base":
        legend = "{} CNN, {} FFN".format(af_name_cnn, af_name_ffn)
    elif net_info.net_type == "ahaf":
        legend = "{}-like AHAF CNN, {}-like AHAF FFN".format(
            af_name_cnn, af_name_ffn
        )
    elif net_info.net_type == "ahaf_ffn":
        legend = "{} CNN, {}-like AHAF FFN".format(
            af_name_cnn, af_name_ffn
        )
    elif net_info.net_type == "leaf":
        legend = "{}-like LEAF CNN, {}-like LEAF FFN".format(
            af_name_cnn, af_name_ffn
        )
    elif net_info.net_type == "leaf_ffn":
        legend = "{} CNN, {}-like LEAF FFN".format(
            af_name_cnn, af_name_ffn
        )
    elif net_info.net_type == "fuzzy_ffn":
        legend = "{} CNN, {}-like Fuzzy FFN".format(
            af_name_cnn, af_name_ffn
        )
    else:
        raise ValueError("Network type is not supported")

    if net_info.fine_tuned:
        legend = legend + ", fine-tuned"

    if net_info.dspu4:
        legend = legend + ", 2SPU-4"

    if net_info.p24sl:
        legend = legend + ", slow LEAF {p2,p4} update"

    return legend


def get_short_af_name(orig: str) -> str:
    if orig == "Tanh":
        return "tanh"
    elif orig == "Sigmoid":
        return "σ-fn"
    else:
        return orig


def get_legend_short(
        net_info: Union[Tuple, NetInfo], omit_af_names: bool = False,
        include_opt: bool = False
) -> str:
    if not isinstance(net_info, NetInfo):
        net_info = NetInfo(*net_info)

    af_name_ffn = net_info.af_name

    if net_info.af_name_cnn is None:
        af_name_cnn = af_name_ffn
    else:
        af_name_cnn = net_info.af_name_cnn

    af_name_cnn = get_short_af_name(af_name_cnn)
    af_name_ffn = get_short_af_name(af_name_ffn)

    if net_info.net_type == "base":
        net_type_str = "Base"
    elif net_info.net_type == "ahaf":
        net_type_str = "AHAF"
    elif net_info.net_type == "ahaf_ffn":
        net_type_str = "AHAF FFN"
    elif net_info.net_type == "leaf":
        net_type_str = "LEAF"
    elif net_info.net_type == "leaf_ffn":
        net_type_str = "LEAF FFN"
    elif net_info.net_type == "fuzzy_ffn":
        net_type_str = "Fuzzy"
    else:
        raise ValueError("Network type is not supported")

    if omit_af_names:
        legend = net_type_str
    else:
        legend = f"{net_type_str}, {af_name_cnn}, {af_name_ffn}"

    if net_info.opt_name == 'adam':
        opt_name_str = 'ADAM'
    elif net_info.opt_name == 'rmsprop':
        opt_name_str = 'RMSprop'
    else:
        raise ValueError("Optimizer is not supported")

    if include_opt:
        legend = f"{legend}, {opt_name_str}"

    if net_info.fine_tuned:
        legend = legend + ", tuned"

    if net_info.dspu4:
        legend = legend + ", DSPT"

    if net_info.p24sl:
        legend = legend + ", P24Sl"

    return legend


def analyze_network(
        net_info: Tuple, omit_af_names: bool = False, include_opt: bool = False
):
    file_path = get_file_name_stat(*net_info)
    results = load_results(file_path)
    base_legend = get_legend_short(net_info, omit_af_names, include_opt)

    acc = []
    loss = []

    for r in results:
        acc.append(float(r["test_acc"]) * 100.0)
        loss.append(float(r["train_loss_mean"]))

    return base_legend, acc, loss


def plot_networks(
        fig, nets: Sequence[Union[Tuple, NetInfo]],
        bw=False, omit_af_names=False, include_opt=False
) -> bool:
    acc_legends = []
    loss_legends = []

    monochrome = (
            cycler('linestyle', ['-', '--', ':', '-.'])
            * cycler('color', ['black', 'grey'])
            * cycler('marker', ['None'])
    )

    gs = plt.GridSpec(1, 2)

    acc_fig = fig.add_subplot(gs[0, 0])
    #acc_loc = plticker.LinearLocator(numticks=10)
    #acc_fig.yaxis.set_major_locator(acc_loc)
    acc_fig.set_xlabel('epoch')
    acc_fig.set_ylabel('test accuracy, %')
    acc_fig.grid()
    if bw:
        acc_fig.set_prop_cycle(monochrome)

    loss_fig = fig.add_subplot(gs[0, 1])
    #loss_loc = plticker.LinearLocator(numticks=10)
    #loss_fig.yaxis.set_major_locator(loss_loc)
    loss_fig.set_xlabel('epoch')
    loss_fig.set_ylabel('training loss')
    loss_fig.grid()
    if bw:
        loss_fig.set_prop_cycle(monochrome)

    net_processed = 0

    for net in nets:
        try:
            base_legend, acc, loss = analyze_network(
                net, omit_af_names, include_opt
            )
        except FileNotFoundError:
            continue
        except Exception as e:
            print("Exception: {}, skipped".format(e))
            continue

        net_processed += 1
        n_epochs = len(acc)
        end_ep = net.epoch
        start_ep = end_ep - n_epochs

        x = tuple(range(start_ep, end_ep))

        acc_legends.append(
            base_legend
        )
        loss_legends.append(
            base_legend
        )
        acc_fig.plot(x, acc)
        loss_fig.plot(x, loss)

    acc_fig.legend(acc_legends)
    loss_fig.legend(loss_legends)

    return net_processed > 0


def visualize(
        net_name: str, ds_name: str, net_group: str,
        nets: Sequence[Union[Tuple, NetInfo]], base_title=None,
        bw: bool = False, omit_af_names: bool = False,
        include_opt: bool = False
):
    fig = plt.figure(tight_layout=True, figsize=(6, 3))
    if base_title is not None:
        title = "{}, test accuracy and training loss".format(base_title)
        fig.suptitle(title)

    success = plot_networks(fig, nets, bw, omit_af_names, include_opt)
    if success:
        #plt.show()
        plt.savefig(get_file_name_stat_img(net_name, ds_name, net_group,
                                           nets[0].epoch, nets[0].patched,
                                           nets[0].fine_tuned))

    plt.close(fig)


def main():
    net_names = ["LeNet-5", "KerasNet"]
    ds_names = ["F-MNIST", "CIFAR-10"]
    net_ds_combinations = itertools.product(net_names, ds_names)
    opt = "adam"

    omit_all_captions = True

    for n, ds in net_ds_combinations:
        relu_comparison = [
            NetInfo(n, "base", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "ReLU", 100, patched=False, fine_tuned=False, dspu4=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
        ]

        visualize(
            n, ds, f"relu_{opt}", relu_comparison,
            None if omit_all_captions else f"{n} on {ds} - ReLU-like AFs",
            bw=False, omit_af_names=True
        )

        leaf_relu_unstable_cmp = [
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name='adam'),
            NetInfo(n, "ahaf", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name='rmsprop'),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name='rmsprop'),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name='rmsprop'),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name='adam'),
        ]

        visualize(
            n, ds, f"leaf_relu_stability", leaf_relu_unstable_cmp,
            None if omit_all_captions else "Learning rate effect on ReLU-like LEAFs",
            bw=True, omit_af_names=True, include_opt=True
        )

        silu_comparison = [
            NetInfo(n, "base", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "SiLU", 100, patched=False, fine_tuned=False, dspu4=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
        ]

        visualize(
            n, ds, f"silu_{opt}", silu_comparison,
            None if omit_all_captions else f"{n} on {ds} - SiLU-like AFs",
            bw=False, omit_af_names=True
        )

        leaf_silu_unstable_cmp = [
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name='adam'),
            NetInfo(n, "ahaf", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name='rmsprop'),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name='rmsprop'),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name='rmsprop'),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name='adam'),
        ]

        visualize(
            n, ds, f"leaf_silu_stability", leaf_silu_unstable_cmp,
            None if omit_all_captions else "Learning rate effect on SiLU-like LEAFs",
            bw=True, omit_af_names=True, include_opt=True
        )

        tanh_comparison = [
            NetInfo(n, "base", ds, "Tanh", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "leaf", ds, "Tanh", 100, patched=False, fine_tuned=False, dspu4=False, p24sl=False, opt_name=opt),
            #NetInfo(n, "leaf", ds, "Tanh", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=False, opt_name=opt),
            NetInfo(n, "leaf", ds, "Tanh", 100, patched=False, fine_tuned=False, dspu4=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "Tanh", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
        ]

        visualize(
            n, ds, f"tanh_{opt}", tanh_comparison,
            None if omit_all_captions else f"{n} on {ds} - Tanh-like AFs",
            bw=True, omit_af_names=True
        )

        sigmoid_comparison = [
            NetInfo(n, "base", ds, "Sigmoid", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "leaf", ds, "Sigmoid", 100, patched=False, fine_tuned=False, dspu4=False, p24sl=False, opt_name=opt),
            #NetInfo(n, "leaf", ds, "Sigmoid", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=False, opt_name=opt),
            NetInfo(n, "leaf", ds, "Sigmoid", 100, patched=False, fine_tuned=False, dspu4=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "Sigmoid", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
        ]

        visualize(
            n, ds, f"sigmoid_{opt}", sigmoid_comparison,
            None if omit_all_captions else f"{n} on {ds} - Sigmoid-like AFs",
            bw=True, omit_af_names=True
        )


if __name__ == "__main__":
    main()
