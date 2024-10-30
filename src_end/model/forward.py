from typing import Optional, Literal, Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src_end.model.parameterization import Readout
from src_end.nn.equiv_layers import EquivLayerNorm
from src_end.nn.layers import DenseLayer, MLP
from src_end.ops import scatter_center


class AffineReadout(Readout):
    def __init__(
            self,
            hn_dim: Tuple[int, int],
            num_node_features: int,
            parameterization: Literal["pinned", "pinned-exp"] = "pinned",
            num_layers: int = 2,
            zero_cog: bool = True,
            delta: float = 1e-2,
            norm_anisotropic: float = 100.,
            layer_norm: bool = False,
            use_v_norm: bool = True
    ) -> None:
        super(AffineReadout, self).__init__()

        sdim, vdim = hn_dim

        if use_v_norm:
            in_dim = sdim + vdim
        else:
            in_dim = sdim

        self.net_h = MLP(in_dim=in_dim,
                         out_dim=num_node_features * 2 + 1,  # one sigma per feature + one sigma per position vector
                         h_dim=sdim,
                         n=num_layers,
                         activation=nn.SiLU(),
                         last_linear=True)
        self.net_pos = DenseLayer(in_features=vdim, out_features=1 + 3, bias=False)  # no bias

        self.num_node_features = num_node_features
        self.parameterization = parameterization
        self.zero_cog = zero_cog
        self.use_v_norm = use_v_norm

        self.register_buffer("delta", torch.as_tensor(delta))
        self.register_buffer("norm_anisotropic", torch.as_tensor(norm_anisotropic))

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
    ) -> Sequence[torch.Tensor]:

        if self.equiv_norm:
            s, v = self.equiv_norm.forward(states, index)
        else:
            s, v = states["s"], states["v"]

        t_index = t[index]

        # ------ scalar part ------
        if self.use_v_norm:
            v_norm = torch.linalg.vector_norm(v, dim=1)
            in_h = torch.cat([s, v_norm], dim=1)
        else:
            in_h = s

        out_h = self.net_h(in_h)
        mean_h, sigma_h, sigma_pos = torch.split(out_h,
                                                 [self.num_node_features, self.num_node_features, 1],
                                                 dim=-1)
        mean_h = (1. - t_index) * (h + t_index * mean_h)

        if self.parameterization == "pinned-exp":
            sigma_h = (self.delta ** (1. - t_index)) * torch.exp(sigma_h * (t_index * (1. - t_index)))
        else:
            sigma_h = (self.delta ** (1. - t_index)) * (F.softplus(sigma_h) ** (t_index * (1. - t_index)))

        # ------ vectorial part ------
        out_pos = self.net_pos(v)
        mean_pos, U_pos = torch.split(out_pos,
                                      [1, 3],
                                      dim=-1)

        mean_pos = mean_pos.squeeze()
        mean_pos = (1. - t_index) * (pos + t_index * mean_pos)

        if self.parameterization == "pinned-exp":
            sigma_pos = (self.delta ** (1. - t_index)) * torch.exp(sigma_pos * (t_index * (1. - t_index)))
        else:
            sigma_pos = (self.delta ** (1. - t_index)) * (F.softplus(sigma_pos) ** (t_index * (1. - t_index)))

        sigma_pos = torch.diag_embed(sigma_pos.expand(-1, 3))

        assert U_pos.shape == sigma_pos.shape

        U_pos = sigma_pos + (1. - t_index[..., None]) * t_index[
            ..., None] * U_pos / self.norm_anisotropic

        if self.zero_cog:
            mean_pos = scatter_center(mean_pos, index=index)

        return mean_h, sigma_h, mean_pos, U_pos
