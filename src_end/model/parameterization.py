import abc
from typing import Optional, Sequence

import torch
import torch.nn as nn

from src_end.nn.encoder import EquivEncoder


class Readout(nn.Module, abc.ABC):
    def forward(self,
                t: torch.Tensor,
                states: dict[str, torch.Tensor],
                h: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                index: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Sequence[torch.Tensor]:
        raise NotImplementedError


class EquivariantParameterization(nn.Module):
    def __init__(
            self,
            encoder: EquivEncoder,
            readout: Readout,
    ):
        super(EquivariantParameterization, self).__init__()
        self.encoder = encoder
        self.readout = readout

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.readout.reset_parameters()

    def forward(self,
                t: torch.Tensor,
                h: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                index: torch.Tensor,
                context: Optional[torch.Tensor] = None):
        states = self.encoder.forward(t=t,
                                      h=h,
                                      pos=pos,
                                      edge_index=edge_index,
                                      index=index,
                                      context=context)
        return self.readout.forward(t,
                                    states,
                                    h=h,
                                    pos=pos,
                                    edge_index=edge_index,
                                    index=index,
                                    context=context)
