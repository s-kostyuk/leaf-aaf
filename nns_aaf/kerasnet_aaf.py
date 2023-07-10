from typing import Optional, List

import torch.nn

from adaptive_afs import AfDefinition, af_build
from .cnn_aaf_base import CnnAafBase


class KerasNetAaf(CnnAafBase):
    """
    KerasNet - CNN implementation evaluated in arXiv 1801.09403 **but** with
    optional support of adaptive activation functions (AHAF, LEAF, F-Neuron
    Activation). The model is based on the example CNN implementation from
    Keras 1.x: git.io/JuHV0.

    Architecture:

    - 2D convolution 32 x (3,3) with (1,1) padding
    - Conv Activation Function
    - 2D convolution 32 x (3,3) w/o padding
    - Conv Activation Function
    - max pooling (2,2)
    - dropout 25%
    - 2D convolution 64 x (3,3) with (1,1) padding
    - Conv Activation Function
    - 2D convolution 64 x (3,3) w/o padding
    - FFN Activation Function
    - max pooling (2,2)
    - dropout 25%
    - fully connected, out_features = 512
    - FFN Activation Function
    - dropout 50%
    - fully connected, out_features = 10
    - softmax activation

    """

    def __init__(
            self, *, flavor='MNIST',
            af_conv: Optional[AfDefinition] = None,
            af_fc: Optional[AfDefinition] = None
    ):
        super().__init__(flavor=flavor, af_conv=af_conv, af_fc=af_fc)

        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=32,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        self.act1 = af_build(
            self._af_def_conv,
            in_dims=(self.conv1.out_channels, *self._act1_img_dims)
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv1.out_channels, out_channels=32,
            kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=True
        )
        self.act2 = af_build(
            self._af_def_conv,
            in_dims=(self.conv2.out_channels, *self._act2_img_dims)
        )

        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop3 = torch.nn.Dropout2d(p=0.25)

        self.conv4 = torch.nn.Conv2d(
            in_channels=self.conv2.out_channels, out_channels=64,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        self.act4 = af_build(
            self._af_def_conv,
            in_dims=(self.conv4.out_channels, *self._act4_img_dims)
        )

        self.conv5 = torch.nn.Conv2d(
            in_channels=self.conv4.out_channels, out_channels=64,
            kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=True
        )
        self.act5 = af_build(
            self._af_def_conv,
            in_dims=(self.conv5.out_channels, *self._act5_img_dims),
        )

        self.pool6 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop6 = torch.nn.Dropout2d(p=0.25)

        self._flatter = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc7 = torch.nn.Linear(
            in_features=self._fc7_in_features,
            out_features=self._fc8_out_features, bias=True
        )

        self.act7 = af_build(self._af_def_fc, in_dims=(self._fc8_out_features,))

        self.drop7 = torch.nn.Dropout(p=0.2)

        self.fc8 = torch.nn.Linear(
            in_features=self._fc8_out_features, out_features=10, bias=True
        )

        # softmax is embedded in pytorch's loss function

        self._sequence = [
            self.conv1, self.act1, self.conv2, self.act2,
            self.pool3, self.drop3,
            self.conv4, self.act4, self.conv5, self.act5,
            self.pool6, self.drop6,
            self._flatter,
            self.fc7, self.act7, self.drop7,
            self.fc8
        ]

    def _init_mnist_dims(self):
        self._image_channels = 1
        self._fc7_in_features = 5 * 5 * 64
        self._fc8_out_features = 512
        self._act1_img_dims = (28, 28)
        self._act2_img_dims = (26, 26)
        self._act4_img_dims = (13, 13)
        self._act5_img_dims = (11, 11)

    def _init_cifar_dims(self):
        self._image_channels = 3
        self._fc7_in_features = 6 * 6 * 64
        self._fc8_out_features = 512
        self._act1_img_dims = (32, 32)
        self._act2_img_dims = (30, 30)
        self._act4_img_dims = (15, 15)
        self._act5_img_dims = (13, 13)

    @property
    def activations(self):
        return [
            self.act1, self.act2, self.act4, self.act5, self.act7
        ]

    @property
    def activation_params(self) -> List[torch.nn.Parameter]:
        params = []

        for act in self.activations:
            if isinstance(act, torch.nn.Module):
                params.extend(act.parameters())

        return params
