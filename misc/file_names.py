from typing import Optional


def _normalize_net_name(name: str) -> str:
    if name == "KerasNet":
        name = "kerasnet"
    elif name == "LeNet-5":
        name = "lenet5"

    return name


def _normalize_ds_name(name: str) -> str:
    if name == "CIFAR-10":
        name = "cifar10"
    elif name == "F-MNIST":
        name = "f-mnist"

    return name


def _format_net_variant(net_name: str, ds_name: str) -> str:
    if ds_name:
        net_name = f"{net_name}_{ds_name}"

    return net_name


def _format_net_af(af_name: str, af_name_cnn: Optional[str]) -> str:
    if af_name_cnn and af_name != af_name_cnn:
        af_name = "{}_{}".format(af_name, af_name_cnn)

    return af_name


def _get_file_name_net_base(
        file_type: str,
        net_name: str, net_type: str, ds_name: str, af_name: str,
        epoch: int, patched: bool, fine_tuned: bool,
        af_name_cnn: Optional[str],
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    patched_str = "patched_" if patched else ""
    fine_tuned_str = "tuned_" if fine_tuned else ""

    net_name = _normalize_net_name(net_name)
    ds_name = _normalize_ds_name(ds_name)
    net_variant = _format_net_variant(net_name, ds_name)
    opt_part = f"{opt_name}_" if opt_name else ""
    train_alg = "dspu4_" if dspu4 else ""
    train_lr = "p24sl_" if p24sl else ""

    if file_type == "stat":
        extension = "csv"
    elif file_type == "aaf_img":
        extension = "svg"
    elif file_type == "args":
        extension = "json"
    else:
        extension = "bin"

    af_name = _format_net_af(af_name, af_name_cnn)

    file_path = f"runs/{net_variant}_{af_name}_" \
                f"{patched_str}{fine_tuned_str}{net_type}_" \
                f"{opt_part}" \
                f"{train_alg}{train_lr}{epoch}ep_{file_type}.{extension}"

    return file_path


def get_file_name_net(
        net_name: str, net_type: str, ds_name: str, af_name: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    return _get_file_name_net_base(
        "net", net_name, net_type, ds_name, af_name, epoch,
        patched, fine_tuned, af_name_cnn, dspu4, p24sl, opt_name
    )


def get_file_name_opt(
        net_name: str, net_type: str, ds_name: str, af_name: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    return _get_file_name_net_base(
        "opt", net_name, net_type, ds_name, af_name, epoch,
        patched, fine_tuned, af_name_cnn, dspu4, p24sl, opt_name
    )


def get_file_name_stat(
        net_name: str, net_type: str, ds_name: str, af_name: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    return _get_file_name_net_base(
        "stat", net_name, net_type, ds_name, af_name, epoch,
        patched, fine_tuned, af_name_cnn, dspu4, p24sl, opt_name
    )


def _get_file_name_net_summary(
        file_type: str,
        net_name: str, ds_name, net_group: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False
) -> str:
    if file_type == "stat_img":
        extension = "svg"
    else:
        extension = "csv"

    patched_str = "_patched" if patched else ""
    fine_tuned_str = "_tuned" if fine_tuned else ""
    net_name = _normalize_net_name(net_name)
    ds_name = _normalize_ds_name(ds_name)
    net_variant = _format_net_variant(net_name, ds_name)

    file_path = f"runs/{net_variant}{patched_str}{fine_tuned_str}_" \
                f"summary_{net_group}_{epoch}ep_{file_type}.{extension}"

    return file_path


def get_file_name_stat_img(
        net_name: str, ds_name, net_group: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False
) -> str:
    return _get_file_name_net_summary(
        "stat_img", net_name, ds_name, net_group, epoch, patched, fine_tuned
    )


def get_file_name_stat_table(
        net_name: str, ds_name, net_group: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False
) -> str:
    return _get_file_name_net_summary(
        "stat_table", net_name, ds_name, net_group, epoch, patched, fine_tuned
    )


def get_file_name_aaf_img(
        net_name: str, net_type: str, ds_name: str, af_name: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    return _get_file_name_net_base(
        "aaf_img", net_name, net_type, ds_name, af_name, epoch,
        patched, fine_tuned, af_name_cnn, dspu4, p24sl, opt_name
    )


def get_file_name_checkp(
        net_name: str, net_type: str, ds_name: str, af_name: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    return _get_file_name_net_base(
        "checkp", net_name, net_type, ds_name, af_name, epoch,
        patched, fine_tuned, af_name_cnn, dspu4, p24sl, opt_name
    )


def get_file_name_train_args(
        net_name: str, net_type: str, ds_name: str, af_name: str, epoch: int,
        patched: bool = False, fine_tuned: bool = False,
        af_name_cnn: Optional[str] = None,
        dspu4: bool = False, p24sl: bool = False, opt_name: str = 'rmsprop'
) -> str:
    return _get_file_name_net_base(
        "args", net_name, net_type, ds_name, af_name, epoch,
        patched, fine_tuned, af_name_cnn, dspu4, p24sl, opt_name
    )
