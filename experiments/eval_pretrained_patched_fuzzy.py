#!/usr/bin/env python3

from eval_pretrained_common import eval_variant


def main():
    af_names = ("Tanh", "Sigmoid")
    net_name = "KerasNet"
    ds_name = "CIFAR-10"

    for af in af_names:
        eval_variant(
            net_name, "fuzzy_ffn", ds_name, af_name=af, start_ep=100,
            patched=True
        )


if __name__ == "__main__":
    main()
