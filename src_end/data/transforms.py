import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import one_hot, dense_to_sparse

from src_end.ops import center


@functional_transform('one_hot')
class OneHot(BaseTransform):
    def __init__(
            self,
            values: list[int],
            key: str = "h",
            scale: float = 1.0,
            noise_std: float = 0.0,
            dtype: torch.dtype = torch.get_default_dtype()
    ) -> None:
        self.mapping = {v: i for (i, v) in enumerate(values)}
        self.key = key
        self.dtype = dtype
        self.noise_std = noise_std
        self.scale = scale

    def forward(self, data: Data) -> Data:
        data_key = getattr(data, self.key)
        assert data_key.ndim == 1

        x = torch.as_tensor([self.mapping[xi.item()] for xi in data_key])
        x = self.scale * one_hot(x, num_classes=len(self.mapping)).to(self.dtype)

        if self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std

        setattr(data, self.key, x)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mapping})'


@functional_transform('fully_connected')
class FullyConnected(BaseTransform):
    def __init__(
            self,
            key: str = "edge_index",
    ) -> None:
        self.key = key

    def forward(self, data: Data) -> Data:
        n = len(data.pos)
        fc_graph = torch.ones(n, n) - torch.eye(n)
        fc_edges, _ = dense_to_sparse(fc_graph)

        setattr(data, self.key, fc_edges)

        return data


@functional_transform('zero_cog')
class ZeroCoG(BaseTransform):
    def __init__(
            self,
            key: str = "pos",
    ) -> None:
        self.key = key

    def forward(self, data: Data) -> Data:
        pos = getattr(data, self.key)
        centered_pos = center(pos)
        setattr(data, self.key, centered_pos)

        return data


@functional_transform('composition')
class Composition(BaseTransform):
    def __init__(
            self,
            key_in: str = "h",
            key_out: str = "context",
            dtype: torch.dtype = torch.get_default_dtype()
    ) -> None:
        self.key_in = key_in
        self.key_out = key_out
        self.dtype = dtype

    def forward(self, data: Data) -> Data:
        data_key_in = getattr(data, self.key_in)
        assert data_key_in.ndim == 2
        one_hot = (data.x > 0.0).long()
        composition = torch.sum(one_hot, dim=0, keepdim=True)

        setattr(data, self.key_out, composition)
        return data


@functional_transform('rename')
class Rename(BaseTransform):
    def __init__(
            self,
            key_in: str = "fp_ob",
            key_out: str = "context",
    ) -> None:
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Data) -> Data:
        data_key_in = getattr(data, self.key_in)
        setattr(data, self.key_out, data_key_in)
        return data
