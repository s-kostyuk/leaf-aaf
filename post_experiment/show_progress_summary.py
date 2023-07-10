#!/usr/bin/env python3

import itertools
import csv

from typing import Sequence, Tuple, List, NamedTuple, Union, Generator, Dict
from decimal import Decimal

from misc import ProgressElement
from misc import NetInfo
from misc import get_file_name_stat, get_file_name_stat_table


def load_results(file_path: str) -> List[ProgressElement]:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_network(net_info: NetInfo) -> Tuple[int, float, float]:
    """
    TBD

    :param net_info: TBD
    :return: best_epoch, best_accuracy, avg_it_sec
    """
    file_path = get_file_name_stat(*net_info)
    results = load_results(file_path)

    max_acc = -1.0
    max_pos = -1
    pos = 0
    total_duration = 0
    # best_duration = -1.0
    pos_offset = net_info.epoch - len(results)

    for el in results:
        pos += 1
        acc = Decimal(el["test_acc"]) * 100
        total_duration += Decimal(el["duration"])

        if acc > max_acc:
            max_acc = acc
            max_pos = pos + pos_offset
            # best_duration = total_duration

    avg_it_time = total_duration / len(results)
    return max_pos, max_acc, round(avg_it_time, 2)


class SummaryItem(NamedTuple):
    net_type: str
    af_cnn: str
    af_ffn: str
    best_acc: float
    best_ep: int
    avg_it_sec: float
    tuned: bool
    dspu4: bool
    p24sl: bool


class SuperSummKey(NamedTuple):
    net: str
    net_type: str
    act: str
    tuned: bool
    dspu4: bool
    p24sl: bool


class SuperSummSubItem(NamedTuple):
    best_acc: float
    best_ep: int


class SuperSummItem(NamedTuple):
    fmnist: SuperSummSubItem
    cifar10: SuperSummSubItem


SuperSumm = Dict[SuperSummKey, SuperSummItem]


def gather_results(nets: Sequence[Union[Tuple, NetInfo]]) -> List[SummaryItem]:
    results = []

    for net in nets:
        if not isinstance(net, NetInfo):
            net = NetInfo(*net)
        try:
            best_ep, best_acc, duration = analyze_network(net)
        except FileNotFoundError:
            continue
        except Exception as e:
            print("Exception: {}, skipped".format(e))
            continue

        net_af_cnn = net.af_name_cnn if net.af_name_cnn else net.af_name
        net_af_ffn = net.af_name

        results.append(
            SummaryItem(net.net_type, net_af_cnn, net_af_ffn, best_acc, best_ep,
                        duration, net.fine_tuned, net.dspu4, net.p24sl)
        )

    return results


def prettify_net_type_short(net_type: str, fine_tuned: bool = False) -> str:
    if net_type == "base":
        net_type = "Base"
    elif net_type == "ahaf":
        net_type = "AHAF"
    elif net_type == "ahaf_ffn":
        net_type = "AHAF FFN"
    elif net_type == "leaf":
        net_type = "LEAF"
    elif net_type == "leaf_ffn":
        net_type = "LEAF FFN"
    elif net_type == "fuzzy_ffn":
        net_type = "Fuzzy"
    else:
        raise ValueError("Network type is not supported")

    if fine_tuned:
        net_type = net_type + " tuned"

    return net_type


def prettify_net_type_long(net_name: str, net_type: str, fine_tuned: bool = False) -> str:
    if net_type == "base":
        net_type = f"Base {net_name}"
    elif net_type == "ahaf":
        net_type = f"{net_name} w/ AHAF"
    elif net_type == "ahaf_ffn":
        net_type = f"{net_name} w/ AHAF FFN"
    elif net_type == "leaf":
        net_type = f"{net_name} w/ LEAF"
    elif net_type == "leaf_ffn":
        net_type = f"{net_name} w/ LEAF FFN"
    elif net_type == "fuzzy_ffn":
        net_type = f"{net_name} w/ Fuzzy FFN"
    else:
        raise ValueError("Network type is not supported")

    if fine_tuned:
        net_type = net_type + " fine-tuned"

    return net_type


def prettify_result(el: SummaryItem) -> Tuple:
    net_type = prettify_net_type_short(el.net_type, el.tuned)

    if el.dspu4 and el.p24sl:
        train_str = 'DSPT, P24Sl'
    elif el.dspu4:
        train_str = 'DSPT'
    elif el.p24sl:
        train_str = 'P24Sl'
    else:
        train_str = 'Stand.'

    return (
        net_type, el.af_cnn, el.af_ffn, train_str,
        el.best_acc, el.best_ep, el.avg_it_sec
    )


def prettify_results(
        results: Sequence[SummaryItem]
) -> Generator[Tuple, None, None]:
    for el in results:
        yield prettify_result(el)


def save_results_as_csv(results: List[SummaryItem], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ("Type", "CNN AF", "FFN AF", "Tr. Alg.", "Accuracy, %", "Epoch", "It. time, s")
        )
        writer.writerows(prettify_results(results))


def summarize(
        net_name: str, ds_name: str, net_group: str,
        nets: Sequence[Union[Tuple, NetInfo]]
) -> List[SummaryItem]:
    results = gather_results(nets)

    if results:
        save_results_as_csv(results, get_file_name_stat_table(
            net_name, ds_name, net_group,
            nets[0].epoch, nets[0].patched, nets[0].fine_tuned
        ))

    return results


def results_to_tex(results: SuperSumm, file_name: str, alt_style: bool = False):
    header1 = """
\\begin{table}[htbp]
\t\\caption{Best test set accuracy, up to 100 epochs}
\t\\label{table:tab1}
\t\\begin{tabular}{llllcccc}
\t\t\\toprule
\t\t& & & & \\multicolumn{2}{c}{Fashion-MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\
\t\t\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}
\t\tNetwork & Activ. & Init. & Proc. & Acc. & Epoch & Acc. & Epoch \\\\
\t\t\\midrule"""

    line_template1 = "\n\t\t{} & {} & {} & {} & {:.2f}\\% & {} & {:.2f}\\% & {} \\\\"

    footer1 = """
\t\t\\bottomrule
\t\\end{tabular}
\\end{table}
"""

    header2 = """
\\begin{table}[htbp]
\t\\caption{Best test set accuracy, up to 100 epochs}
\t\\label{table:tab1}
\t\\begin{tabular}{|p{30pt}|p{15pt}|p{15pt}|p{20pt}||p{20pt}|p{10pt}||p{20pt}|p{10pt}|}
\t\t\\hline
\t\t& & & & \\multicolumn{2}{c||}{\\textbf{F-MNIST}} & \\multicolumn{2}{c|}{\\textbf{CIFAR-10}} \\\\
\t\t\\cline{5-6}\\cline{7-8}
\t\t\\textbf{Network} & \\textbf{Activ.} & \\textbf{Init.} & \\textbf{Proc.} & \\textbf{Acc.,\\%} & \\textbf{Ep.} & \\textbf{Acc.,\\%} & \\textbf{Ep.} \\\\
\t\t\\hline"""

    line_template2 = "\n\t\t{} & {} & {} & {} & {:.2f}\\% & {} & {:.2f}\\% & {} \\\\ \\hline"

    footer2 = """
\t\\end{tabular}
\\end{table}
"""

    if alt_style:
        header = header2
        line_template = line_template2
        footer = footer2
    else:
        header = header1
        line_template = line_template1
        footer = footer1

    with open(file_name, 'w') as f:
        f.write(header)

        for item in results:
            net = item.net
            var = item.net_type

            if item.dspu4 and item.p24sl:
                trainer = "DSPT, P24Sl"
            elif item.dspu4:
                trainer = "DSPT"
            elif item.p24sl:
                trainer = "P24Sl"
            else:
                trainer = "Classic"

            base_act = item.act

            act_str = var.upper() if var != "base" else base_act
            init_str = base_act if var != "base" else "N/A"

            item_value = results[item]

            line_str = line_template.format(
                net, act_str, init_str, trainer,
                item_value.fmnist.best_acc, item_value.fmnist.best_ep,
                item_value.cifar10.best_acc, item_value.cifar10.best_ep
            )
            f.write(line_str)

        f.write(footer)


def results_to_tex_sep_tables(net: str, results: SuperSumm, file_name: str, alt_style: bool = False):
    header1 = """
\\begin{table}[htbp]
\t\\caption{Best test set accuracy, up to 100 epochs}
\t\\label{table:tab1}
\t\\begin{tabular}{lllcccc}
\t\t\\toprule
\t\t& & & \\multicolumn{2}{c}{Fashion-MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\
\t\t\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}
\t\tActiv. & Init. & Proc. & Acc. & Epoch & Acc. & Epoch \\\\
\t\t\\midrule"""

    line_template1 = "\n\t\t{} & {} & {} & {:.2f}\\% & {} & {:.2f}\\% & {} \\\\"

    footer1 = """
\t\t\\bottomrule
\t\\end{tabular}
\\end{table}
"""

    header2 = """
\\begin{table}[htbp]
\t\\caption{Best test set accuracy, up to 100 epochs}
\t\\label{table:tab1}
\t\\begin{tabular}{|p{20pt}|p{20pt}|p{45pt}|p{25pt}|p{10pt}|p{25pt}|p{10pt}|}
\t\t\\hline
\t\t& & & \\multicolumn{2}{c|}{\\textbf{F-MNIST}} & \\multicolumn{2}{c|}{\\textbf{CIFAR-10}} \\\\
\t\t\\cline{4-5}\\cline{6-7}
\t\t\\textbf{Activ.} & \\textbf{Init.} & \\textbf{Procedure} & \\textbf{Acc.,\\%} & \\textbf{Ep.} & \\textbf{Acc.,\\%} & \\textbf{Ep.} \\\\
\t\t\\hline"""

    line_template2 = "\n\t\t{} & {} & {} & {:.2f}\\% & {} & {:.2f}\\% & {} \\\\ \\hline"

    footer2 = """
\t\\end{tabular}
\\end{table}
"""

    if alt_style:
        header = header2
        line_template = line_template2
        footer = footer2
    else:
        header = header1
        line_template = line_template1
        footer = footer1

    with open(file_name, 'w') as f:
        f.write(header)

        for item in results:
            if net != item.net:
                continue

            var = item.net_type

            if item.dspu4 and item.p24sl:
                trainer = "DSPT, P24Sl"
            elif item.dspu4:
                trainer = "DSPT"
            elif item.p24sl:
                trainer = "P24Sl"
            else:
                trainer = "Classic"

            base_act = item.act if item.act != 'Sigmoid' else 'Sigm.'

            act_str = var.upper() if var != "base" else base_act
            init_str = base_act if var != "base" else "N/A"

            item_value = results[item]

            line_str = line_template.format(
                act_str, init_str, trainer,
                item_value.fmnist.best_acc, item_value.fmnist.best_ep,
                item_value.cifar10.best_acc, item_value.cifar10.best_ep
            )
            f.write(line_str)

        f.write(footer)


def extend_super_summary(
        net: str, ds: str, results: List[SummaryItem], ss: SuperSumm
):
    for result in results:
        if result.af_cnn == result.af_ffn:
            base_af = result.af_cnn
        else:
            base_af = f'{result.af_cnn}, {result.af_ffn}'

        key = SuperSummKey(net, result.net_type, base_af, result.tuned, result.dspu4, result.p24sl)
        if key not in ss:
            # initialize this combination
            ss[key] = SuperSummItem(
                SuperSummSubItem(-1.0, -1),
                SuperSummSubItem(-1.0, -1),
            )

        if ds == 'F-MNIST':
            ss[key] = SuperSummItem(
                SuperSummSubItem(result.best_acc, result.best_ep),
                ss[key].cifar10
            )
        elif ds == 'CIFAR-10':
            ss[key] = SuperSummItem(
                ss[key].fmnist,
                SuperSummSubItem(result.best_acc, result.best_ep)
            )


def main():
    net_names = ["LeNet-5", "KerasNet"]
    ds_names = ["F-MNIST", "CIFAR-10"]
    net_ds_combinations = itertools.product(net_names, ds_names)
    opt = "adam"
    all_lin_uns = {}  # type: SuperSumm
    all_bou_fns = {}  # type: SuperSumm

    for n, ds in net_ds_combinations:
        nets_vs_ahaf = [
            NetInfo(n, "base", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "ReLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "ReLU", 100, patched=False, fine_tuned=False, dspu4=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "ReLU", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
            NetInfo(n, "base", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "SiLU", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "ahaf", ds, "SiLU", 100, patched=False, fine_tuned=False, dspu4=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "SiLU", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
        ]

        lin_uns_results = summarize(n, ds, f"lin_un_{opt}", nets_vs_ahaf)
        extend_super_summary(n, ds, lin_uns_results, all_lin_uns)

        nets_vs_fuzzy = [
            NetInfo(n, "base", ds, "Tanh", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "leaf", ds, "Tanh", 100, patched=False, fine_tuned=False, dspu4=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "Tanh", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
            NetInfo(n, "base", ds, "Sigmoid", 100, patched=False, fine_tuned=False, opt_name=opt),
            NetInfo(n, "leaf", ds, "Sigmoid", 100, patched=False, fine_tuned=False, dspu4=False, p24sl=True, opt_name=opt),
            NetInfo(n, "leaf", ds, "Sigmoid", 100, patched=False, fine_tuned=False, dspu4=True, p24sl=True, opt_name=opt),
        ]

        bou_fns_results = summarize(n, ds, f"bou_fn_{opt}", nets_vs_fuzzy)
        extend_super_summary(n, ds, bou_fns_results, all_bou_fns)

    results_to_tex_sep_tables(
        'LeNet-5', all_lin_uns,
        f'runs/lenet5_all_summary_lin_un_{opt}_100ep_stat_table.tex',
        alt_style=True
    )
    results_to_tex_sep_tables(
        'KerasNet', all_lin_uns,
        f'runs/kerasnet_all_summary_lin_un_{opt}_100ep_stat_table.tex',
        alt_style=True
    )
    results_to_tex_sep_tables(
        'LeNet-5', all_bou_fns,
        f'runs/lenet5_all_summary_bou_fn_{opt}_100ep_stat_table.tex',
        alt_style=True
    )
    results_to_tex_sep_tables(
        'KerasNet', all_bou_fns,
        f'runs/kerasnet_all_summary_bou_fn_{opt}_100ep_stat_table.tex',
        alt_style=True
    )


if __name__ == "__main__":
    main()
