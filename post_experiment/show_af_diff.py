import contextlib
import itertools
from typing import Optional, NamedTuple

import torch
import torch.nn.functional
from matplotlib import pyplot as plt
from cycler import cycler

from adaptive_afs import af_build, AfDefinition
from adaptive_afs.trad import tanh_manual, silu_manual


class ErrorHolder(NamedTuple):
    min_error: float
    max_error: float


def estimate_error(
        orig_fn, drv_fn, left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    n_points = 100000

    range_len = right - left
    step = range_len / n_points
    eps = step / 100

    with torch.no_grad():
        x = torch.arange(start=left, end=right + eps, step=step)

        y = orig_fn(x)
        y_hat = drv_fn(x)

        errors = torch.square(y - y_hat)
        max_error = torch.max(errors)
        min_error = torch.min(errors)

        x_view = x.cpu().numpy()
        err_view = errors.cpu().numpy()

        monochrome = cycler('color', ['black'])
        plt.rcParams['axes.prop_cycle'] = monochrome

        plt.xlabel("Input, x")
        plt.ylabel("Error, E=Δ^2")
        plt.title(f"min={min_error.item()},max={max_error.item()}")

        plt.plot(x_view, err_view)

        if img_path is None:
            plt.show()
        else:
            plt.savefig(img_path, dpi=300, format='svg')
            plt.close()

    return ErrorHolder(min_error.item(), max_error.item())


def estimate_err_manual_silu(
        left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    orig_fn = torch.nn.functional.silu
    drv_fn = silu_manual

    return estimate_error(orig_fn, drv_fn, left, right, img_path)


def estimate_err_manual_tanh(
        left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    orig_fn = torch.tanh
    drv_fn = tanh_manual

    return estimate_error(orig_fn, drv_fn, left, right, img_path)


def estimate_err_aaf(
        af_def: AfDefinition,
        left=-4.0, right=+4.0, img_path: Optional[str] = None
) -> ErrorHolder:
    orig_fn = af_build(
        AfDefinition(af_def.af_base, AfDefinition.AfType.TRAD)
    )
    drv_fn = af_build(af_def)

    return estimate_error(orig_fn, drv_fn, left, right, img_path)


def estimate_all(dev_name: str, prec_name: str):
    estimate_err_manual_silu(
        -15.0, +15.0,
        img_path=f"runs/af_diff_manual_silu_{dev_name}_{prec_name}.svg"
    )
    estimate_err_manual_tanh(
        -15.0, +15.0,
        img_path=f"runs/af_diff_manual_tanh_{dev_name}_{prec_name}.svg"
    )

    af_defs_fuzz = [
        AfDefinition(af_base="Tanh", af_type=AfDefinition.AfType.ADA_FUZZ,
                     af_interval=AfDefinition.AfInterval(-12.0, +12.0, 768))
    ]

    for ff in af_defs_fuzz:
        img_path = f"runs/af_diff_fuzz_{ff.af_base}_{dev_name}_{prec_name}.svg"
        estimate_err_aaf(ff, ff.interval.start, ff.interval.end, img_path)

    af_names_ahaf = ["ReLU", "SiLU"]

    for afn in af_names_ahaf:
        img_path = f"runs/af_diff_ahaf_{afn}_{dev_name}_{prec_name}.svg"
        af_def = AfDefinition(af_base=afn, af_type=AfDefinition.AfType.ADA_AHAF)
        estimate_err_aaf(af_def, -15.0, +15.0, img_path)

    af_names_leaf = ["ReLU", "SiLU", "Tanh", "Sigmoid"]

    for afn in af_names_leaf:
        img_path = f"runs/af_diff_leaf_{afn}_{dev_name}_{prec_name}.svg"
        af_def = AfDefinition(af_base=afn, af_type=AfDefinition.AfType.ADA_LEAF)
        estimate_err_aaf(af_def, -15.0, +15.0, img_path)


@contextlib.contextmanager
def precision(name: str):
    prev_dtype = torch.get_default_dtype()

    if name == "float16":
        new_dtype = torch.float16
    elif name == "float64":
        new_dtype = torch.float64
    else:
        new_dtype = torch.float32

    torch.set_default_dtype(new_dtype)

    try:
        yield
    finally:
        torch.set_default_dtype(prev_dtype)


def main():
    devices = ["cpu", "cuda"]
    precisions = ["float16", "float32", "float64"]

    for dev, prec in itertools.product(devices, precisions):
        if dev == "cpu" and prec == "float16":
            # Skip, not implemented in PyTorch
            continue

        with torch.device(dev):
            with precision(prec):
                estimate_all(dev, prec)


if __name__ == "__main__":
    main()
