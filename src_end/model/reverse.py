from typing import Literal, Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src_end.model.parameterization import Readout
from src_end.nn.equiv_layers import EquivLayerNorm
from src_end.nn.layers import DenseLayer, MLP
from src_end.ops import scatter_center


class DataPointReadout(Readout):
    def __init__(
            self,
            hn_dim: Tuple[int, int],
            out_dim: int,
            parameterization: Literal["residual-pos", "residual-time-pos"],
            zero_cog: bool = True,
            num_layers: int = 2,
            softmax_h: bool = False,
            scale_softmax_h: float = 1.,
            layer_norm: bool = False,
    ) -> None:
        super(DataPointReadout, self).__init__()
        sdim, vdim = hn_dim

        self.net_h = MLP(in_dim=sdim,
                         out_dim=out_dim,
                         h_dim=sdim,
                         n=num_layers,
                         activation=nn.SiLU(),
                         last_linear=True)
        self.net_pos = DenseLayer(in_features=vdim, out_features=1, bias=False)

        self.parameterization = parameterization
        self.zero_cog = zero_cog

        self.softmax_h = softmax_h
        self.scale_softmax_h = scale_softmax_h
        self.equiv_norm = EquivLayerNorm(hn_dim) if layer_norm else False

        self.reset_parameters()

    def reset_parameters(self):
        self.net_h.reset_parameters()
        self.net_pos.reset_parameters()

    def forward(
            self,
            t: torch.Tensor,
            states: dict[str, torch.Tensor],
            h: torch.Tensor,
            pos: torch.Tensor,
            edge_index: Optional[torch.Tensor],
            index: torch.Tensor,
            context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.equiv_norm:
            s, v = self.equiv_norm.forward(states, index)
        else:
            s, v = states["s"], states["v"]

        out_h = self.net_h(s)
        if self.softmax_h:
            out_h = F.softmax(out_h, dim=-1) * self.scale_softmax_h

        out_pos = self.net_pos(v).squeeze()

        match self.parameterization:
            case "residual-pos":
                out_pos += pos

            case "residual-time-pos":
                t_index = t[index]
                out_pos = pos + t_index * out_pos

        if self.zero_cog:
            out_pos = scatter_center(out_pos, index=index)

        return out_h, out_pos


class DirectReadout(nn.Module):
    def __init__(
            self,
            hn_dim: Tuple[int, int],
            out_dim: int,
            zero_cog: bool = True,
            num_layers: int = 2,
            layer_norm: bool = False,
    ) -> None:
        super(DirectReadout, self).__init__()
        sdim, vdim = hn_dim

        self.net_h = MLP(in_dim=sdim,
                         out_dim=2 * out_dim,
                         h_dim=sdim,
                         n=num_layers,
                         activation=nn.SiLU(),
                         last_linear=True)

        self.net_pos = DenseLayer(in_features=vdim, out_features=2, bias=False)
        self.zero_cog = zero_cog

        self.equiv_norm = EquivLayerNorm(hn_dim) if layer_norm else False

        self.reset_parameters()

    def reset_parameters(self):
        self.net_h.reset_parameters()
        self.net_pos.reset_parameters()

    def forward(
            self,
            t: torch.Tensor,
            states: dict[str, torch.Tensor],
            h: torch.Tensor,
            pos: torch.Tensor,
            edge_index: Optional[torch.Tensor],
            index: torch.Tensor,
            context: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

        if self.equiv_norm:
            s, v = self.equiv_norm.forward(states, index)
        else:
            s, v = states["s"], states["v"]

        out_h = self.net_h(s)
        out_pos = self.net_pos(v)

        ode_h, score_h = torch.chunk(out_h, chunks=2, dim=-1)
        ode_pos, score_pos = out_pos[..., 0], out_pos[..., 1]

        return (ode_h, score_h), (ode_pos, score_pos)
