import math


# Based on https://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self):
        self._num_points = 0  # type: int
        self._old_mean = None
        self._old_variance = None
        self._new_mean = None
        self._new_variance = None

    def clear(self):
        self._num_points = 0

    def push(self, x: float):
        self._num_points += 1

        if self._num_points == 1:
            self._old_mean = self._new_mean = x
            self._old_variance = 0.0
        else:
            self._new_mean = self._old_mean + (x - self._old_mean) / self._num_points
            self._new_variance = self._old_variance + (x - self._old_mean) * (x - self._new_mean)

            self._old_mean = self._new_mean
            self._old_variance = self._new_variance

    @property
    def num_datapoints(self) -> int:
        return self._num_points

    @property
    def mean(self) -> float:
        return self._new_mean if self._num_points > 0 else 0

    @property
    def variance(self) -> float:
        return self._new_variance / (self._num_points - 1) if self._num_points > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)
