import torch.nn


class TriangularMembF(torch.nn.Module):
    def __init__(self, radius: float, center: float):
        super().__init__()

        self._radius = radius
        self._center = center
        self._left = center - radius
        self._right = center + radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Center the inputs
        x = x - self._center

        # Scale inputs to the -1.0..+1.0 range
        x = x / self._radius

        # Use absolute values
        x = torch.absolute(x)

        # Compute output: y = 1 - abs(x)
        x = 1 - x

        # Drop outliers (drop negative values after all the operations above)
        x = torch.clip(x, 0.0)

        return x

    def __repr__(self):
        return "triangle: {},{},{}".format(self._left, self._center, self._right)
