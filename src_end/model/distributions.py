import torch
from torch import nn

from src_end.ops import scatter_center


class GaussianDistribution(nn.Module):
    def __init__(self,
                 dim_h: int,
                 dim_pos: int = 3,
                 zero_cog: bool = True):
        super(GaussianDistribution, self).__init__()
        self.dim_h = dim_h
        self.dim_pos = dim_pos
        self.zero_cog = zero_cog

    def sample(self,
               index: torch.Tensor):
        sample_h = torch.randn((len(index), self.dim_h), device=index.device)
        sample_pos = torch.randn((len(index), self.dim_pos), device=index.device)
        if self.zero_cog:
            sample_pos = scatter_center(sample_pos, index=index)

        return sample_h, sample_pos
