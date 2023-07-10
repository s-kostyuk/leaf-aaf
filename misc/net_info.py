from typing import NamedTuple, Optional


class NetInfo(NamedTuple):
    net_name: str
    net_type: str
    ds_name: str
    af_name: str
    epoch: int
    patched: bool = False,
    fine_tuned: bool = False
    af_name_cnn: Optional[str] = None
    dspu4: bool = False
    p24sl: bool = False
    opt_name: str = "rmsprop"
