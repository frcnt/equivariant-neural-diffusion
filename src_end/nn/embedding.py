import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from src_end.nn.layers import MLP


class FourierEmbedding(nn.Module):
    """
    Random Fourier features (sine and cosine expansion).
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std: float = 1.0,
            trainable: bool = False,
    ):
        super(FourierEmbedding, self).__init__()
        assert (out_features % 2) == 0
        weight = torch.normal(mean=torch.zeros(out_features // 2, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x):
        x = F.linear(x, self.weight)
        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)
        x = torch.cat((cos_features, sin_features), dim=1)

        return x


def cosine_cutoff(edge_distances: torch.Tensor, cutoff: float):
    return torch.where(
        edge_distances < cutoff,
        .5 * (torch.cos(torch.pi * edge_distances / cutoff) + 1.),
        torch.tensor(0.0, device=edge_distances.device, dtype=edge_distances.dtype),
    )


class EdgeEmbedding(nn.Module):
    def __init__(self,
                 num_rbf_features: int = 64,
                 max_distance: float = 25.0,
                 trainable: bool = False,
                 norm: bool = True,
                 cutoff: bool = True):
        super().__init__()

        self.norm = norm
        self.n_rbf_features = num_rbf_features
        self.max_distance = max_distance
        self.cutoff = cutoff

        self.register_buffer("delta", torch.tensor(max_distance / num_rbf_features))
        offsets = torch.linspace(start=0., end=max_distance, steps=num_rbf_features).unsqueeze(0)
        if trainable:
            self.offsets = nn.Parameter(offsets)
        else:
            self.register_buffer("offsets", offsets)

    def forward(self,
                positions: torch.Tensor,
                edge_index: torch.Tensor,
                norm: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = self.norm if norm is None else norm
        dest, source = edge_index

        vectors = (
                positions[dest] - positions[source]
        )  # (n_edges, 3) vector (i - > j)

        distances = torch.sqrt(
            torch.sum(vectors ** 2, dim=-1, keepdim=True) + 1e-6
        )  # (n_edges, 1)
        d = self.featurize_distances(distances)

        cos = F.cosine_similarity(positions[dest], positions[source], dim=-1).unsqueeze(1)

        if norm:
            vectors = vectors / (distances + 1.0)

        return d, cos, vectors  # (n_edges, 1), (n_edges, 3)

    def featurize_distances(self, distances: torch.Tensor):
        distances = torch.clamp(distances, 0., self.max_distance)
        features = torch.exp((-((distances - self.offsets) ** 2)) / self.delta)

        if self.cutoff:
            features = features * cosine_cutoff(distances, cutoff=self.max_distance)

        return features


class CompositionEmbedding(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 h_dim: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Parameter(torch.randn(in_dim, h_dim[0]))
        self.mlp = MLP(in_dim=h_dim[0] * in_dim, out_dim=out_dim,
                       h_dim=h_dim,
                       n=len(h_dim),
                       activation=nn.SiLU())

    def forward(self, composition: torch.Tensor):
        norm_composition = composition / torch.sum(composition, dim=-1, keepdim=True)
        embedded_composition = norm_composition[..., None] * self.embedding[None, ...]
        embedded_composition = embedded_composition.flatten(start_dim=1)

        return self.mlp(embedded_composition)

    def reset_parameters(self):
        self.mlp.reset_parameters()
