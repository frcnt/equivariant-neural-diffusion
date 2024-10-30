import math
from typing import Union, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import zeros_, kaiming_normal_


class DenseLayer(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: Union[Callable, nn.Module] = None,
            weight_init: Callable = kaiming_normal_,
            bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 h_dim: int | list[int],
                 n: int,
                 activation: Union[Callable, nn.Module] = None,
                 last_linear: bool = True):
        super().__init__()

        if isinstance(h_dim, int):
            all_in_dim = [in_dim] + [h_dim] * (n - 1)
            all_out_dim = [h_dim] * (n - 1) + [out_dim]
            all_in_out_dim = list(zip(all_in_dim, all_out_dim))
        else:
            all_in_dim = [in_dim] + h_dim
            all_out_dim = h_dim + [out_dim]
            all_in_out_dim = list(zip(all_in_dim, all_out_dim))

        layers = []
        for in_d, out_d in all_in_out_dim[:-1]:
            layers += [DenseLayer(in_d, out_d, activation=activation)]

        in_d, out_d = all_in_dim[-1], all_out_dim[-1]
        if last_linear:
            layers += [DenseLayer(in_d, out_d)]
        else:
            layers += [DenseLayer(in_d, out_d, activation=activation)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def reset_parameters(self):
        for layer in self.mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
