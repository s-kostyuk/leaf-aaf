from typing import Optional, List

import torch.nn

from adaptive_afs import AfDefinition


class CnnAafBase(torch.nn.Module):
    def __init__(
            self, *, flavor='MNIST',
            af_conv: Optional[AfDefinition] = None,
            af_fc: Optional[AfDefinition] = None
    ):
        super().__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_mnist_dims()
        elif flavor == 'CIFAR10':
            self._init_cifar_dims()
        else:
            raise NotImplemented("Other flavors of LeNet-5 are not supported")

        if af_conv is None:
            # Use ReLU in the convolutional layers by default
            af_conv = AfDefinition(
                af_base="ReLU", af_type=AfDefinition.AfType.TRAD,
                af_interval=None
            )

        self._af_def_conv = af_conv

        if af_fc is None:
            # Use ReLU in the fully connected layers by default
            af_fc = AfDefinition(
                af_base="ReLU", af_type=AfDefinition.AfType.TRAD,
                af_interval=None
            )

        self._af_def_fc = af_fc

        self._sequence = []

    def _init_mnist_dims(self):
        raise NotImplementedError(
            "The MNIST support shall be implemented on the child class level"
        )

    def _init_cifar_dims(self):
        raise NotImplementedError(
            "The CIFAR-10 support shall be implemented on the child class level"
        )

    def forward(self, x):
        for mod in self._sequence:
            x = mod(x)

        return x

    @property
    def activations(self):
        raise NotImplementedError(
            "The list of activations shall be specified on the child class "
            "level"
        )

    @property
    def activation_params(self) -> List[torch.nn.Parameter]:
        params = []

        for act in self.activations:
            if isinstance(act, torch.nn.Module):
                params.extend(act.parameters())

        return params
