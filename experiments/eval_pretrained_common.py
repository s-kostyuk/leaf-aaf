#!/usr/bin/env python3

import torch
import torch.nn
import torch.utils.data
import torchinfo

from experiments.common import get_device, get_dataset
from misc import get_file_name_net, create_net


def eval_variant(
        net_name: str, net_type: str, ds_name: str, af_name: str, start_ep: int,
        *, patched: bool = False
):
    batch_size = 64
    rand_seed = 42

    print(
        "Loading pre-trained {} {} network with {}{} activation "
        "on the {} dataset after {} epochs.".format(
            net_type, net_name, af_name, "" if net_type == "base" else "-like",
            ds_name, start_ep
        )
    )

    path_net = get_file_name_net(
       net_name, net_type, ds_name, af_name, start_ep, patched
    )

    dev = get_device()
    torch.manual_seed(rand_seed)

    _, test_set = get_dataset(ds_name, augment=True)
    input_size = (batch_size, 3, 32, 32)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, num_workers=4
    )

    net = create_net(net_name, net_type, ds_name, af_name)
    net.to(device=dev)
    torchinfo.summary(net, input_size=input_size, device=dev)

    missing, unexpected = net.load_state_dict(
        torch.load(path_net), strict=True
    )

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    with torch.no_grad():
        net.eval()
        test_total = 0
        test_correct = 0

        for batch in test_loader:
            x = batch[0].to(dev)
            y = batch[1].to(dev)
            y_hat = net(x)
            _, pred = torch.max(y_hat.data, 1)
            test_total += y.size(0)
            test_correct += torch.eq(pred, y).sum().item()

        print("Epoch: {}. Test set accuracy: {:.2%}".format(
            start_ep, test_correct / test_total
        ))

