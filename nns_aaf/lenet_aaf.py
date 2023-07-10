from typing import Optional, List

import torch.nn

from adaptive_afs import AfDefinition, af_build
from .cnn_aaf_base import CnnAafBase


class LeNetAaf(CnnAafBase):
    """
    Implementation of LeNet-5 with the support of AHAF, LEAF and the
    F-Neuron Activation. The network structure follows arXiv 1801.09403.
    """

    def __init__(
            self, *, flavor='MNIST',
            af_conv: Optional[AfDefinition] = None,
            af_fc: Optional[AfDefinition] = None
    ):
        super().__init__(flavor=flavor, af_conv=af_conv, af_fc=af_fc)

        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=20,
            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.act1 = af_build(
            self._af_def_conv,
            in_dims=(self.conv1.out_channels, *self._act1_img_dims)
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = torch.nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )
        self.act2 = af_build(
            self._af_def_conv,
            in_dims=(self.conv2.out_channels, *self._act2_img_dims)
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self._flatter = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc3 = torch.nn.Linear(
            in_features=self._fc3_in_features, out_features=500, bias=True
        )
        self.act3 = af_build(self._af_def_fc, in_dims=(self.fc3.out_features,))

        self.fc4 = torch.nn.Linear(
            in_features=500, out_features=10, bias=False
        )
        # SoftMax is embedded into the Cross Entropy loss function.

        self._sequence = [
            self.conv1, self.act1, self.pool1,
            self.conv2, self.act2, self.pool2,
            self._flatter,
            self.fc3, self.act3,
            self.fc4
        ]

    def _init_mnist_dims(self):
        self._image_channels = 1
        self._fc3_in_features = 4 * 4 * 50
        self._act1_img_dims = (24, 24)
        self._act2_img_dims = (8, 8)

    def _init_cifar_dims(self):
        self._image_channels = 3
        self._fc3_in_features = 5 * 5 * 50
        self._act1_img_dims = (28, 28)
        self._act2_img_dims = (10, 10)

    @property
    def activations(self):
        return [
            self.act1, self.act2, self.act3
        ]

