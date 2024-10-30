from typing import Optional, Callable

import numpy as np
import torch
import torch.utils.data as torchdata
from torch_geometric.data import Data
from torch_geometric.io import fs


class Dataset(torchdata.Dataset):
    def __init__(
            self,
            path: str,
            transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.data = self.load(path)

    @staticmethod
    def load(path):
        return fs.torch_load(path)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = data if self.transform is None else self.transform(data)

        return data

    def __len__(self):
        return len(self.data)


class SampleDataset(torchdata.Dataset):
    def __init__(
            self,
            empirical_distribution: np.ndarray,
            n_samples: int = 10_000,
            transform: Optional[Callable] = None,
            seed: int = 42,
    ) -> None:
        self.empirical_distribution = empirical_distribution

        rng = np.random.RandomState(seed)
        self.num_atoms = rng.choice(len(empirical_distribution), n_samples, p=empirical_distribution)

        self.transform = transform

    def __getitem__(self, idx):
        n = self.num_atoms[idx]
        data = Data(
            pos=torch.randn(n, 3),  # NOTE, necessary for proper edge/idx creation
            h=torch.LongTensor([6] * n),  # NOTE: default to carbon
        )
        data = data if self.transform is None else self.transform(data)

        return data

    def __len__(self):
        return len(self.num_atoms)
