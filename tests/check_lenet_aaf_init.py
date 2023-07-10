#!/usr/bin/env python3

import torchinfo

from nns_aaf import LeNetAaf, AfDefinition


def print_act_functions(net: LeNetAaf):
    print(net.activations)


def main():
    nn_defs = [
        (None, None),  # expected: all ReLU
        (AfDefinition("SiLU"), AfDefinition("SiLU")),  # expected: all SiLU

        # expected: SiLU in CNN, adaptive SiLU in FFN, range: -10...+10
        (
            AfDefinition("SiLU"),
            AfDefinition(
                "SiLU", AfDefinition.AfType.ADA_AHAF
            )
        ),

        # expected: SiLU in CNN, HardTanh in FFN, range: -10...+10
        (
            AfDefinition("SiLU"),
            AfDefinition(
                "HardTanh", AfDefinition.AfType.TRAD,
                AfDefinition.AfInterval(-10.0, +10.0)
            )
        ),

        # expected: SiLU in CNN, Fuzzy HardTanh in FFN, range: -10...+10
        (
            AfDefinition("SiLU"),
            AfDefinition(
                "HardTanh", AfDefinition.AfType.ADA_FUZZ,
                AfDefinition.AfInterval(-10.0, +10.0, n_segments=12)
            )
        ),

        # expected:
        # AHAF as SiLU in CNN,
        # Fuzzy Sigmoid in FFN, range: -3...+3,
        (
            AfDefinition("SiLU", AfDefinition.AfType.ADA_AHAF),
            AfDefinition(
                "Sigmoid", AfDefinition.AfType.ADA_FUZZ,
                AfDefinition.AfInterval(-3.0, +3.0, n_segments=12)
            )
        ),

        # expected: SiLU in CNN, AHAF as SiLU in FFN
        (
            AfDefinition("SiLU", AfDefinition.AfType.TRAD),
            AfDefinition("SiLU", AfDefinition.AfType.ADA_AHAF),
        ),

        # expected: all LEAF as SiLU
        (
            AfDefinition("SiLU", AfDefinition.AfType.ADA_LEAF),
            AfDefinition("SiLU", AfDefinition.AfType.ADA_LEAF),
        )
    ]

    batch_size = 64
    image_dim = (1, 28, 28)
    input_size = (batch_size, *image_dim)

    for d in nn_defs:
        net = LeNetAaf(flavor='F-MNIST', af_conv=d[0], af_fc=d[1])
        torchinfo.summary(net, input_size=input_size)
        print_act_functions(net)


if __name__ == "__main__":
    main()
