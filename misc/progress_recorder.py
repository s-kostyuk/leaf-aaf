import time

from typing import TypedDict, List, Mapping
from csv import DictWriter


class ProgressElement(TypedDict):
    epoch: int
    train_loss_mean: float
    train_loss_var: float
    test_acc: float
    lr: str
    duration: float


class ProgressRecorder(object):
    def __init__(self):
        self._els = []  # type: List[Mapping]
        self._ep_start = None
        self._ep_delta = None

    def start_ep(self):
        self._ep_start = time.time()

    def end_ep(self):
        ep_end = time.time()

        if self._ep_start is None:
            self._ep_delta = None
        else:
            self._ep_delta = ep_end - self._ep_start

        self._ep_start = None

    def push_ep(
            self, epoch: int, train_loss_mean: float, train_loss_var: float,
            test_acc: float, lr: str
    ):
        self._els.append(
            ProgressElement(
                epoch=epoch,
                train_loss_mean=train_loss_mean,
                train_loss_var=train_loss_var,
                test_acc=test_acc, lr=lr,
                duration=self._ep_delta
            )
        )
        self._ep_delta = None

    def save_as_csv(self, path: str):
        fields = ProgressElement.__annotations__.keys()

        with open(path, 'w') as f:
            writer = DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(self._els)
