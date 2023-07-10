from typing import Tuple

from ..fuzzy.f_neuron_act import FNeuronAct


def af_build_fuzzy(
        af_base: str, af_start: float, af_end: float, n_segments: int,
        in_dims: Tuple[int, ...] = (1,)
) -> FNeuronAct:
    init_f = FNeuronAct.get_init_f_by_name(af_base)

    return FNeuronAct(
        af_start, af_end, n_segments,
        init_f=init_f, input_dim=in_dims
    )
