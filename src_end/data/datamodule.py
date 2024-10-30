from pathlib import Path
from typing import Optional

import numpy as np
import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src_end.data.dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self,
                 transform: T.BaseTransform,
                 train_path: str | Path,
                 val_path: str | Path,
                 train_batch_size: int,
                 val_batch_size: int,
                 num_val_subset: Optional[int] = 500,
                 test_path: Optional[str | Path] = None,
                 test_batch_size: Optional[int] = None,
                 num_test_subset: Optional[int] = 10000,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 subset_seed: int = 42, ):
        super().__init__()

        self.train_dataset = Dataset(path=train_path,
                                     transform=transform)

        val_dataset = Dataset(path=val_path, transform=transform)
        if isinstance(num_val_subset, int) and num_val_subset < len(val_dataset):
            val_dataset = self.get_random_subset(val_dataset, num_val_subset, seed=subset_seed)
        self.val_dataset = val_dataset

        if test_path:
            test_dataset = Dataset(path=test_path, transform=transform)
            if isinstance(num_test_subset, int) and num_test_subset > -1 and num_test_subset < len(test_dataset):
                test_dataset = self.get_random_subset(test_dataset, num_test_subset, seed=subset_seed)
            self.test_dataset = test_dataset
        else:
            self.test_dataset = None

        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.hparams.train_batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    @staticmethod
    def get_random_subset(dataset, subset_size, seed):
        rnd = np.random.RandomState(seed=seed)
        indices = rnd.permutation(np.arange(subset_size))
        subset = Subset(dataset, indices=indices)

        return subset
