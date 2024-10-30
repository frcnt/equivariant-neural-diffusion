import dataclasses
from typing import Optional

import torch
import torch.nn as nn

from src_end.nn.embedding import EdgeEmbedding, CompositionEmbedding, FourierEmbedding
from src_end.nn.equiv_layers import EQGATBackbone
from src_end.nn.layers import DenseLayer, MLP
from src_end.ops import scatter_center


@dataclasses.dataclass
class EquivEncoderHParams:
    num_layers: int
    num_node_features: int
    hidden_scalar_dim: int
    hidden_vector_dim: int
    hidden_edge_dim: int
    vector_aggr: str = "mean"
    zero_cog: bool = True
    num_edge_features: int = 1
    max_distance: float = 25.
    cutoff: bool = True
    num_fourier_features: int = 0

    use_context: bool = False
    context_type: str = None
    context_where: str = "nodes"
    num_context_features: int = None
    context_hidden_dims: list[int] = None

    def __post_init__(self):
        if self.use_context:
            assert self.context_hidden_dims is not None, "'context_hidden_dims' should be defined when 'use_context'=True"
            assert self.num_context_features is not None, "'num_context_features' should be defined when 'use_context'=True"
            assert self.context_where in (
                "nodes", "edges", "mp"), "'context_where' should be defined either 'nodes' or 'edges'"


class EquivEncoder(nn.Module):
    def __init__(
            self,
            hparams: EquivEncoderHParams,
            **kwargs,
    ):
        super(EquivEncoder, self).__init__(**kwargs)
        self.hparams = hparams

        if hparams.num_fourier_features > 0:
            self.time_features = FourierEmbedding(1, hparams.num_fourier_features)
            t_in = hparams.num_fourier_features
        else:
            self.time_features = lambda x: x
            t_in = 1

        self.time_mapping_node = DenseLayer(t_in, self.hparams.hidden_scalar_dim)
        self.time_mapping_edge = DenseLayer(t_in, self.hparams.hidden_edge_dim)

        self.node_mapping = DenseLayer(self.hparams.num_node_features, self.hparams.hidden_scalar_dim)
        self.edge_mapping = DenseLayer(1 + self.hparams.num_edge_features, self.hparams.hidden_edge_dim)

        self.node_time_mapping = DenseLayer(self.hparams.hidden_scalar_dim, self.hparams.hidden_scalar_dim)
        self.edge_time_mapping = DenseLayer(self.hparams.hidden_edge_dim, self.hparams.hidden_edge_dim)

        self.edge_feature = EdgeEmbedding(num_rbf_features=self.hparams.num_edge_features,
                                          max_distance=self.hparams.max_distance,
                                          cutoff=self.hparams.cutoff)

        if self.hparams.use_context:
            if self.hparams.context_where in ("nodes", "mp"):
                out_dim = self.hparams.hidden_scalar_dim
            else:
                out_dim = self.hparams.hidden_edge_dim

            self.time_mapping_context = DenseLayer(t_in, out_dim)
            self.context_time_mapping = DenseLayer(out_dim, out_dim)

            if self.hparams.context_type == "composition":
                self.context_encoder = CompositionEmbedding(in_dim=self.hparams.num_context_features,
                                                            out_dim=out_dim,
                                                            h_dim=self.hparams.context_hidden_dims)
            else:
                self.context_encoder = MLP(in_dim=self.hparams.num_context_features,
                                           out_dim=out_dim,
                                           h_dim=self.hparams.context_hidden_dims,
                                           n=len(self.hparams.context_hidden_dims),
                                           activation=nn.SiLU())

        self.backbone = EQGATBackbone(hn_dim=(self.hparams.hidden_scalar_dim, self.hparams.hidden_vector_dim),
                                      edge_dim=self.hparams.hidden_edge_dim,
                                      d_dim=self.hparams.num_edge_features,
                                      num_layers=self.hparams.num_layers,
                                      vector_aggr=self.hparams.vector_aggr,
                                      use_context=bool(self.hparams.use_context and self.hparams.context_where == "mp"))

        self.reset_parameters()

    def reset_parameters(self):
        self.time_mapping_node.reset_parameters()
        self.time_mapping_edge.reset_parameters()

        self.node_mapping.reset_parameters()
        self.edge_mapping.reset_parameters()

        self.node_time_mapping.reset_parameters()
        self.edge_time_mapping.reset_parameters()

        if self.hparams.use_context:
            self.node_time_mapping.reset_parameters()
            self.edge_time_mapping.reset_parameters()
            self.context_encoder.reset_parameters()

        self.backbone.reset_parameters()

    def forward(self,
                t: torch.Tensor,
                h: torch.Tensor,
                pos: torch.Tensor,
                edge_index: Optional[torch.Tensor],
                index: torch.Tensor,
                context: Optional[torch.Tensor] = None):

        src_idx, _ = edge_index
        t = self.time_features(t)

        t_node = self.time_mapping_node(t)
        t_edge = self.time_mapping_edge(t)

        t_node = t_node[index]
        t_edge = t_edge[index][src_idx]

        node_states_s = self.node_time_mapping(self.node_mapping(h) + t_node)
        node_states_v = pos.new_zeros(
            (*pos.shape, self.hparams.hidden_vector_dim)
        )

        if self.hparams.zero_cog:
            pos = scatter_center(pos, index=index)

        edge_features = self.edge_feature(
            pos, edge_index
        )  # dist, cos, vectors

        edge_states = torch.cat(edge_features[:2], dim=-1)
        edge_states = self.edge_time_mapping(self.edge_mapping(edge_states) + t_edge)

        if self.hparams.use_context:
            assert isinstance(context, torch.Tensor), f"'context' should be 'torch.Tensor' but got '{type(context)}'"
            t_context = self.time_mapping_context(t)
            context_states = self.context_time_mapping(self.context_encoder(context) + t_context)

            if self.hparams.context_where == "nodes":
                node_states_s = node_states_s + context_states[index]
                context = None
            elif self.hparams.context_where == "edges":
                edge_states = edge_states + context_states[index][src_idx]
                context = None
            else:
                context = context_states[index]

        edge_attr = (*edge_features, edge_states)  # (dist, cos, vectors, states)

        out = self.backbone.forward(s=node_states_s,
                                    v=node_states_v,
                                    p=pos,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    index=index,
                                    context=context)

        return out
