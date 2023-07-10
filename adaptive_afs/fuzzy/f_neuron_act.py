from typing import Callable, Tuple

import torch.nn
from torch.nn import functional as F

from .fmf_triangular import TriangularMembF
from .fmf_ramp_left import LeftRampMembF
from .fmf_ramp_right import RightRampMembF


class FNeuronAct(torch.nn.Module):
    @staticmethod
    def ramp_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Initialize member function weights to create a ramp function
        from -1.0 to +1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered,
               ignored by this implementation.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        low = -1.0
        high = +1.0
        range_ = high - low
        step = range_ / (count + 1)
        eps = step / 100
        sample = torch.arange(low, high + eps, step)
        result = torch.empty(*input_dim, len(sample))
        return result.copy_(sample)

    @staticmethod
    def inv_ramp_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Initialize member function weights to create an inverse ramp function
        from +1.0 to -1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered,
               ignored by this implementation.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        return - FNeuronAct.ramp_init(count, input_dim, in_range)

    @staticmethod
    def random_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Random weights initialization, ranging from -1.0 to +1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered,
               ignored by this implementation.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        low = -1.0
        high = +1.0
        out_range = high - low
        return low + torch.rand(*input_dim, count + 2) * out_range

    @staticmethod
    def all_hot_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Constant weights initialization, all membership functions active with
        the same weight of 1.0.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered,
               ignored by this implementation.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        return torch.ones(*input_dim, count + 2)

    @staticmethod
    def _init_as_function(
            orig_function: Callable[[torch.Tensor], torch.Tensor],
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            left: float = -1.0,
            right: float = +1.0
    ) -> torch.Tensor:
        range_ = right - left
        step = range_ / (count + 1)
        eps = step / 100
        sample = torch.arange(left, right + eps, step)
        sample = orig_function(sample)
        result = torch.empty(*input_dim, len(sample))
        return result.copy_(sample)

    @staticmethod
    def tanh_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Weights initialization that creates an activation function that roughly
        corresponds to an approximated tanh function.

        NOTE: The minimum recommended input range is from -5.0 to +5.0 to
        follow tanh with less than 10^-3 error outside the range.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        left, right = in_range
        return FNeuronAct._init_as_function(
            torch.tanh, count, input_dim, left, right
        )

    @staticmethod
    def sigmoid_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Weights initialization that creates an activation function that roughly
        corresponds to an approximated sigmoid function.

        NOTE: The minimum recommended input range is from -10.0 to +10.0 to
        follow sigmoid with less than 10^-3 error outside the range.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        left, right = in_range
        return FNeuronAct._init_as_function(
            torch.sigmoid, count, input_dim, left, right
        )

    @staticmethod
    def hard_sigmoid_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Weights initialization that creates an activation function that exactly
        corresponds to a hard sigmoid.

        NOTE: The minimum recommended input range is from -3.0 to +3.0,
        because the hard sigmoid function is defined on exactly this range.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        left, right = in_range
        return FNeuronAct._init_as_function(
            F.hardsigmoid, count, input_dim, left, right
        )

    @staticmethod
    def hard_tanh_init(
            count: int,
            input_dim: Tuple[int, ...] = (1,),
            in_range: Tuple[float, float] = (-1.0, +1.0)
    ) -> torch.Tensor:
        """
        Weights initialization that creates an activation function that exactly
        corresponds to a hard tanh.

        NOTE: The minimum recommended input range is from -1.0 to +1.0,
        because by default the hard tanh function is defined on exactly this
        range in PyTorch.

        :param count: number of member functions.
        :param input_dim: input data dimensions:
            - scalar ``(1,)`` - by default;
            - vector ``(x,)``;
            - matrix ``(x,y)``;
            - multi-channel image: ``(z,x,y)`` where ``z`` is the number of
              channels.
        :param in_range: the range of input values that should be covered.
        :return: initialized tensor of size ``(z,x,y,count)`` where dimensions
                 ``z`` and ``y`` are optional.
        """
        left, right = in_range
        return FNeuronAct._init_as_function(
            F.hardsigmoid, count, input_dim, left, right
        )

    @classmethod
    def get_init_f_by_name(
            cls, init_f_name: str
    ) -> Callable[[int, Tuple[int, ...], Tuple[float, float]], torch.Tensor]:
        if init_f_name == "Ramp":
            fuzzy_init_f = FNeuronAct.ramp_init
        elif init_f_name == "Random":
            fuzzy_init_f = FNeuronAct.random_init
        elif init_f_name == "Constant":
            fuzzy_init_f = FNeuronAct.all_hot_init
        elif init_f_name == "Tanh":
            fuzzy_init_f = FNeuronAct.tanh_init
        elif init_f_name == "Sigmoid":
            fuzzy_init_f = FNeuronAct.sigmoid_init
        elif init_f_name == "HardSigmoid":
            fuzzy_init_f = FNeuronAct.hard_sigmoid_init
        elif init_f_name == "HardTanh":
            fuzzy_init_f = FNeuronAct.hard_tanh_init
        else:
            raise NotImplementedError(
                "Other initialization functions for fuzzy weights are not "
                "supported."
            )

        return fuzzy_init_f

    def __init__(
            self, left: float, right: float, count: int,
            *, init_f: Callable[[int, Tuple[int, ...]], torch.Tensor] = None,
            input_dim=(1,)
    ):
        super().__init__()
        self._mfs = torch.nn.ModuleList()

        assert left < right
        assert count >= 1

        if init_f is None:
            init_f = self.ramp_init

        self._weights = torch.nn.Parameter(
            init_f(count, input_dim, (left, right))
        )
        self._mf_radius = (right - left) / (count + 1)

        self._mfs.append(
            LeftRampMembF(self._mf_radius, left)
        )

        for i in range(1, count + 1):
            mf_center = left + self._mf_radius * i
            mf = TriangularMembF(self._mf_radius, mf_center)

            self._mfs.append(mf)

        self._mfs.append(
            RightRampMembF(self._mf_radius, right)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [mf.forward(x) for mf in self._mfs]
        x = torch.stack(x, -1)
        x = torch.mul(x, self._weights)
        x = torch.sum(x, -1)
        return x
