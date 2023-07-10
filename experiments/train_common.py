import json
import warnings

from typing import Optional, Callable, List, Union, Iterable, TypedDict, Any, \
    Dict

import torch
import torch.nn
import torch.utils.data
import torchinfo

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from adaptive_afs import LEAF
from experiments.common import get_device, get_dataset
from misc import get_file_name_checkp, get_file_name_stat,\
    get_file_name_train_args

from nns_aaf import KerasNetAaf, LeNetAaf
from misc import RunningStat, ProgressRecorder, create_net


AafNetwork = Union[KerasNetAaf, LeNetAaf]


class CheckPoint(TypedDict):
    net: Dict[str, Any]
    opts: List[Dict[str, Any]]


def _net_train_aaf(net: AafNetwork):
    for p in net.parameters():
        p.requires_grad = False

    for p in net.activation_params:
        p.requires_grad = True


def _net_train_non_aaf(net: AafNetwork):
    for p in net.parameters():
        p.requires_grad = True

    for p in net.activation_params:
        p.requires_grad = False


def _net_train_noop(net: AafNetwork):
    pass


def _net_split_leaf_params(net: AafNetwork):
    aaf_param_ids = set()
    leaf_p24_params = []
    aaf_rest_params = []
    generic_params = []

    for act in net.activations:
        if isinstance(act, LEAF):
            p24_params = (act.p2, act.p4)
            p13_params = (act.p1, act.p3)

            leaf_p24_params.extend(p24_params)
            aaf_rest_params.extend(p13_params)
            aaf_param_ids.update(
                (id(p) for p in act.parameters())
            )
        elif isinstance(act, torch.nn.Module):
            aaf_rest_params.extend(act.parameters())
            aaf_param_ids.update(
                (id(p) for p in act.parameters())
            )

    for p in net.parameters():
        if id(p) not in aaf_param_ids:
            generic_params.append(p)

    return leaf_p24_params, aaf_rest_params, generic_params


def get_opt_by_name(
        opt_name: str, base_lr: float,
        net_params: Iterable[Union[torch.nn.Parameter, Dict]]
) -> torch.optim.Optimizer:
    if opt_name == 'rmsprop':
        opt = torch.optim.RMSprop(
            params=net_params,
            lr=base_lr,
            alpha=0.9,  # default Keras
            momentum=0.0,  # default Keras
            eps=1e-7,  # default Keras
            centered=False  # default Keras
        )
    elif opt_name == 'adam':
        opt = torch.optim.Adam(
            params=net_params,
            lr=base_lr,
        )
    else:
        raise NotImplementedError("Only ADAM and RMSProp supported")

    return opt


def train_variant(
        net_name: str, net_type: str,
        ds_name: str, af_name: str, end_epoch: int = 100, *,
        start_epoch: int = 0, patched: bool = False,
        af_name_cnn: Optional[str] = None,
        param_freezer: Optional[Callable[[AafNetwork], None]] = None,
        save_as_fine_tuned: bool = False,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop',
        seed: int = 42, bs: int = 64, dev_name: str = 'gpu',
        patch_base: bool = False, wandb_enable: bool = False
):
    """
    Initialize, load and train the model for the specified number of epochs.
    Saves the trained network, the optimizer state and the statistics to the
    `./runs` directory.

    :param net_name: name of the model - "KerasNet" or "LeNet-5"
    :param net_type: type of the model - "base", "ahaf", "leaf", "fuzzy_ffn"
    :param ds_name: name of the dataset - "CIFAR-10" or "F-MNIST"
    :param af_name: the initial activation function form name -
                    "ReLU", "SiLU", "Tanh", "Sigmnoid" and so on
    :param end_epoch: stop the training at this epoch
    :param start_epoch: start the training at this epoch
    :param patched: indicates to load the "patched" model that was initially
                    trained with the base activation and then upgraded to an
                    adaptive alternative
    :param af_name_cnn: specify the different initial activation function form
                        for the fully connected layers of the network
    :param param_freezer: a callback function to freeze some parameters in the
                          network before starting the training
    :param save_as_fine_tuned: saves the files with the "tuned_" suffix
    :param dspu4: set to `True` to use the 2SPU-4 training procedure
    :param p24sl: set to `True` to decrease LR for LEAF params p2 and p4
    :param opt_name: set the optimizer: 'adam' or 'rmsprop'
    :param seed: the initial value for RNG
    :param bs: the training data block size
    :param dev_name: training executor device
    :param patch_base: perform in-place patching of the base network
    :param wandb_enable: enable logging to Weights and Biases
    :return: None
    """
    if param_freezer and dspu4:
        raise ValueError(
            "The parameter freezing function and the 2SPU-4 procedure can't be "
            "enabled and used at the same time"
        )

    if wandb_enable and not WANDB_AVAILABLE:
        raise ValueError(
            "The wandb library is not available. Install the library or disable"
            "logging to Weights and Biases in the arguments"
        )

    batch_size = bs
    rand_seed = seed

    dev = get_device(dev_name)
    torch.manual_seed(rand_seed)
    torch.use_deterministic_algorithms(mode=True)

    train_set, test_set = get_dataset(ds_name, augment=True)
    input_size = (batch_size, *train_set[0][0].shape)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, num_workers=4
    )

    net = create_net(
        net_name, net_type, ds_name, af_name, af_name_cnn=af_name_cnn
    )

    error_fn = torch.nn.CrossEntropyLoss()
    net.to(device=dev)
    torchinfo.summary(net, input_size=input_size, device=dev)

    if opt_name == 'rmsprop':
        base_lr = 1e-4
        p24lr = base_lr / 10
    elif opt_name == 'adam':
        base_lr = 0.001
        p24lr = base_lr / 1000
    else:
        raise NotImplementedError("Only ADAM and RMSProp supported")

    if not net_type.startswith("leaf"):
        # Ignore on everything except LEAF
        p24sl = False

    net_params_leaf_p24, net_params_aaf_rest, net_params_non_aaf = _net_split_leaf_params(net)
    opt_params_non_aaf = [
        {'params': net_params_non_aaf}
    ]

    if p24sl:
        print(f"Using a custom learning rate of {p24lr} for LEAF params "
              f"p2 and p4")
        opt_params_aaf = [
            {'params': net_params_aaf_rest},
            {'params': net_params_leaf_p24, 'lr': p24lr}
        ]
    else:
        opt_params_aaf = [
            {'params': [*net_params_aaf_rest, *net_params_leaf_p24]}
        ]

    opt_params_sets: List[List[Dict]]

    if dspu4:
        # Create two different optimizers: one for AAF, one for non-AAF params
        opt_params_sets = [
            opt_params_aaf, net_params_non_aaf
        ]
    else:
        # Create a single optimizer for AAF and non-AAF params
        opt_params_sets = [
            [*opt_params_non_aaf, *opt_params_aaf]
        ]

    opts: List[torch.optim.Optimizer]
    opts = [get_opt_by_name(opt_name, base_lr, ps) for ps in opt_params_sets]

    if start_epoch > 0:
        net_type_to_load = "base" if patch_base else net_type
        strict_load = not patch_base
        dspu4_to_load = False if patch_base else dspu4
        p24sl_to_load = False if patch_base else p24sl

        path_checkp = get_file_name_checkp(
            net_name, net_type_to_load, ds_name, af_name, start_epoch, patched,
            af_name_cnn=af_name_cnn, dspu4=dspu4_to_load,
            p24sl=p24sl_to_load, opt_name=opt_name
        )

        checkp: CheckPoint
        checkp = torch.load(path_checkp)
        net.load_state_dict(checkp['net'], strict=strict_load)

        if ('opts' in checkp) and (not patch_base):
            opt_states = checkp['opts']
            assert len(opts) == len(opt_states)
            for i in range(len(opt_states)):
                opts[i].load_state_dict(opt_states[i])
        else:
            warnings.warn(
                "The old optimizer state is not available{}. Initialized the "
                "optimizer from scratch.".format(
                    " after patching" if (patched or patch_base) else ""
                )
            )

    print(
        "Training the {} {} network with {} in CNN and {} in FFN "
        "on the {} dataset for {} epochs total using the {} training procedure "
        "and the {} optimizer."
        "".format(
            net_type, net_name, af_name if af_name_cnn is None else af_name_cnn,
            af_name, ds_name, end_epoch, "2SPU-4" if dspu4 else "standard",
            opt_name
        )
    )

    # Freeze the parameters if such hook is defined.
    if param_freezer is not None:
        param_freezer(net)

    mb_param_freezers = []  # type: List[Callable[[AafNetwork], None]]
    if dspu4:
        mb_param_freezers.append(_net_train_aaf)
        mb_param_freezers.append(_net_train_non_aaf)
    else:
        mb_param_freezers.append(_net_train_noop)
    assert len(opts) == len(mb_param_freezers)

    progress = ProgressRecorder()

    # TODO: Refactor, pass TypedDict as the function argument
    args_content = {
        "net_name": net_name,
        "net_type": net_type,
        "ds_name": ds_name,
        "af_name": af_name,
        "end_epoch": end_epoch,
        "start_epoch": start_epoch,
        "patched": patched,
        "af_name_cnn": af_name_cnn,
        #"param_freezer": param_freezer,
        "save_as_fine_tuned": save_as_fine_tuned,
        "dspu4": dspu4,
        "p24sl": p24sl,
        "opt_name": opt_name,
        "seed": seed,
        "bs": bs,
        "dev_name": dev_name,
        "patch_base": patch_base
    }

    args_path = get_file_name_train_args(
            net_name, net_type, ds_name, af_name, end_epoch,
            patched or patch_base,
            fine_tuned=save_as_fine_tuned, af_name_cnn=af_name_cnn,
            dspu4=dspu4, p24sl=p24sl, opt_name=opt_name
    )

    if wandb_enable:
        wandb_run_name = args_path.lstrip("runs/")
        wandb_run_name = wandb_run_name.rstrip("_args.json")
        wandb.init(
            project='leaf-cnn', reinit=True, name=wandb_run_name,
            config=args_content, group=f"{net_name}_{ds_name}_{af_name}"
        )
        wandb.watch(net, criterion=error_fn, log_freq=390, log='all')

    with open(args_path, 'w') as f:
        json.dump(args_content, f, indent=2)

    for epoch in range(start_epoch, end_epoch):
        net.train()
        loss_stat = RunningStat()
        progress.start_ep()

        for mb in train_loader:
            x, y = mb[0].to(dev), mb[1].to(dev)
            last_loss_in_mb: float = -1.0

            for mbf, opt in zip(mb_param_freezers, opts):
                mbf(net)

                # The wandb logger does not support `net.forward()`
                y_hat = net(x)
                loss = error_fn(y_hat, target=y)
                last_loss_in_mb = loss.item()

                # Update parameters
                opt.zero_grad()
                loss.backward()
                opt.step()

            loss_stat.push(last_loss_in_mb)

        progress.end_ep()
        net.eval()

        with torch.no_grad():
            test_total = 0
            test_correct = 0

            for batch in test_loader:
                x = batch[0].to(dev)
                y = batch[1].to(dev)
                y_hat = net(x)
                test_loss = error_fn(y_hat, target=y)
                _, pred = torch.max(y_hat.data, 1)
                test_total += y.size(0)
                test_correct += torch.eq(pred, y).sum().item()

            test_acc = test_correct / test_total

            print("Train set loss stat: m={}, var={}".format(
                loss_stat.mean, loss_stat.variance
            ))
            print("Epoch: {}. Test set accuracy: {:.2%}. Test set loss: {:.2}".format(
                    epoch, test_acc, test_loss
            ))
            if wandb_enable:
                wandb.log({
                    'train_loss': loss_stat.mean,
                    'test_loss': test_loss,
                    'test_acc': test_acc}
                )
            progress.push_ep(
                epoch, loss_stat.mean, loss_stat.variance, test_acc,
                lr=' '.join(
                    [str(pg["lr"]) for opt in opts for pg in opt.param_groups]
                )
            )

    progress.save_as_csv(
        get_file_name_stat(
            net_name, net_type, ds_name, af_name, end_epoch,
            patched or patch_base,
            fine_tuned=save_as_fine_tuned, af_name_cnn=af_name_cnn,
            dspu4=dspu4, p24sl=p24sl, opt_name=opt_name
        )
    )

    checkp: CheckPoint
    checkp = {
        'net': net.state_dict(),
        'opts': [opt.state_dict() for opt in opts]
    }

    torch.save(
        checkp,
        get_file_name_checkp(
            net_name, net_type, ds_name, af_name, end_epoch,
            patched or patch_base,
            fine_tuned=save_as_fine_tuned, af_name_cnn=af_name_cnn,
            dspu4=dspu4, p24sl=p24sl, opt_name=opt_name
        )
    )

    if wandb_enable:
        wandb.finish()
