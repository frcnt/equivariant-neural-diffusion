import abc
from typing import Sequence

import torch
import torch.nn as nn


class Volatility(nn.Module, abc.ABC):
    def forward(self, t: torch.Tensor) -> Sequence[torch.Tensor]:
        raise NotImplementedError


class LinearVolatility(Volatility):
    def __init__(self,
                 d: int,
                 gamma_0: float = -10.,
                 gamma_1: float = 10.):
        super(LinearVolatility, self).__init__()
        self.d = d
        self.register_buffer("gamma_0", torch.as_tensor(gamma_0))
        self.register_buffer("gamma_1", torch.as_tensor(gamma_1))

    def forward(self, t: torch.Tensor):
        dg_dt = (self.gamma_1 - self.gamma_0)
        s2 = torch.sigmoid(self.gamma_0 + dg_dt * t)
        g = torch.sqrt(s2 * dg_dt)

        return [g for _ in range(self.d)]
