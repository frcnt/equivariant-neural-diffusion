import os
from pathlib import Path
from typing import Any, Union
from typing import Optional

import ase
import numpy as np
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger

from src_end.data.utils import save_images


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    wandb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            if wandb_logger is not None:
                raise ValueError('More than one WandbLogger was found in the list of loggers')

            wandb_logger = logger

    return wandb_logger


def make_atoms_grid(atoms_lst: list[ase.Atoms]):
    bs = len(atoms_lst)
    nrows = ncols = int(np.ceil(np.sqrt(bs)))

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    for i, ax in enumerate(fig.axes):
        if i >= len(atoms_lst):
            break
        atoms = atoms_lst[i]
        plot_atoms(atoms, ax)
    return fig


class LogSampledAtomsCallback(Callback):
    def __init__(self,
                 dirpath: Union[Path, str],
                 save_atoms: bool = True,
                 num_log_wandb: int = 25,
                 prefix_with_epoch: bool = True):
        self.dirpath = dirpath
        self.save_atoms = save_atoms
        self.num_log_wandb = num_log_wandb
        self.prefix_with_epoch = prefix_with_epoch

        self.atoms_lst: list[ase.Atoms] = ...

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.atoms_lst = []

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_validation_start(trainer, pl_module)

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: list[ase.Atoms],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.atoms_lst.extend(outputs)

    def on_test_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: list[ase.Atoms],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        return self.on_validation_batch_end(trainer,
                                            pl_module,
                                            outputs,
                                            batch,
                                            batch_idx,
                                            dataloader_idx)

    def on_validation_epoch_end(
            self, trainer: Trainer, pl_module: LightningModule
    ) -> None:

        super().on_validation_epoch_end(trainer, pl_module)
        epoch = pl_module.current_epoch

        if self.prefix_with_epoch:
            dirpath = os.path.join(self.dirpath, str(epoch))
        else:
            dirpath = self.dirpath

        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        if self.save_atoms:
            save_path = os.path.join(dirpath, "samples.xyz")
            save_images(self.atoms_lst, filename=save_path)

        if self.num_log_wandb:
            logger: WandbLogger = get_wandb_logger(trainer)

            idx = min(len(self.atoms_lst), self.num_log_wandb)
            fig = make_atoms_grid(self.atoms_lst[-idx:])

            logger.log_image(f'val/images', [fig])
            plt.close(fig)

    def on_test_epoch_end(
            self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        return self.on_validation_epoch_end(trainer, pl_module)
