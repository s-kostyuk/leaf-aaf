from typing import Union, Optional, Tuple

from .af_definition import AfDefinition
from ..cont import AHAF, LEAF
from ..fuzzy import FNeuronAct
from .af_build_fuzzy import af_build_fuzzy
from .af_build_traditional import af_build_traditional, AfTraditional

ActivationFunction = Union[
    AfTraditional, AHAF, LEAF, FNeuronAct
]


def af_build(
        d: AfDefinition, in_dims: Optional[Tuple[int, ...]] = None
) -> ActivationFunction:
    if in_dims is None:
        # Has sense only for adaptive activations
        in_dims = (1,)

    if d.af_type == AfDefinition.AfType.TRAD:
        if d.interval is None:
            return af_build_traditional(d.af_base)
        else:
            return af_build_traditional(
                d.af_base,
                d.interval.start,
                d.interval.end
            )
    elif d.af_type == AfDefinition.AfType.ADA_AHAF:
        return AHAF(size=in_dims, init_as=d.af_base)
    elif d.af_type == AfDefinition.AfType.ADA_LEAF:
        return LEAF(size=in_dims, init_as=d.af_base)
    elif d.af_type == AfDefinition.AfType.ADA_FUZZ:
        return af_build_fuzzy(
            d.af_base,
            d.interval.start, d.interval.end,
            d.interval.n_segments,
            in_dims
        )
    else:
        raise NotImplementedError("The requested AF type is not supported")
