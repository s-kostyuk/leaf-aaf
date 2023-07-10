import torch.nn


class LeftRampMembF(torch.nn.Module):
    """
    -----
         \\
          \\
           \\_______
    """
    def __init__(self, radius: float, center: float):
        super().__init__()

        self._radius = radius
        self._center = center
        self._right = center + radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to the center
        x = x - self._center

        # Scale to the -1.0..1.0 range
        x = x / self._radius

        # Clip the value to create a ramp
        x = torch.clip(x, min=0.0, max=1.0)

        # Invert the value to create a left ramp
        x = 1 - x

        return x

    def __repr__(self):
        return "left: {},{}".format(self._center, self._right)
