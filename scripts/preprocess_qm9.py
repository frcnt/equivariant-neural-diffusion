import os.path
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import ase.io
import fire
import numpy as np
import torch
import tqdm
from torch_geometric.data import Data

from src_end import utils
from src_end.data.utils import read_json, save_json
from src_end.metrics.fp import calc_fp_ob
from src_end.metrics.qm9 import QM9Metrics

log = utils.get_pylogger(__name__)

SEED = 0

URL_GDB9 = "https://ndownloader.figshare.com/files/3195389"
N_GDB9 = 133885

URL_UNCHARACTERIZED = "https://ndownloader.figshare.com/files/3195404"
N_UNCHARACTERIZED = 3054

try:
    from openbabel import pybel, openbabel as ob

    log.info(f"OB is installed.")
    OB_INSTALLED = True
except:
    log.info(f"OB is not installed. Pre-processisng will proceed without it.")
    OB_INSTALLED = False


def read_uncharacterized(fpath: str):
    with open(fpath) as f:
        return [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]


def read_file(datafile, center: bool = True) -> ase.Atoms:
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_xyz = xyz_lines[2:num_atoms + 2]

    symbols, positions = [], []
    for line in mol_xyz:
        symbol, posx, posy, posz, _ = line.replace('*^', 'e').split()
        symbols.append(symbol)
        positions.append([float(posx), float(posy), float(posz)])

    atoms = ase.Atoms(symbols=symbols, positions=positions)
    if center:
        atoms.positions -= atoms.positions.mean(axis=0)

    return atoms


def process_atoms(atoms: ase.Atoms):
    def convert_to_pyg():
        data_dict = dict(h=torch.LongTensor(atoms.numbers),
                         pos=torch.Tensor(atoms.positions))
        if OB_INSTALLED:
            _, fp_ob = calc_fp_ob(atoms)
            data_dict["fp_ob"] = torch.BoolTensor(fp_ob).view(1, -1)

        return Data(**data_dict)

    pyg_data = convert_to_pyg()

    return pyg_data


def preprocess_qm9(
        target_dir: str | Path = "../data/qm9/",
        split_file: Optional[str | Path] = None, ):
    # Download
    download_dir = os.path.join(target_dir, "download")
    os.makedirs(download_dir, exist_ok=True)
    log.info(f"The downloaded files will be placed in '{download_dir}'.")

    fname_gdb9 = os.path.join(download_dir, "dsgdb9nsd.xyz.tar.bz2")
    if not os.path.exists(fname_gdb9):
        log.info(f"Downloading '{URL_GDB9}'...")
        urllib.request.urlretrieve(URL_GDB9, filename=fname_gdb9)
        log.info(f"Done downloading '{URL_GDB9}'...")

    fname_uncharacterized = os.path.join(download_dir, "uncharacterized.txt")
    if not os.path.exists(fname_uncharacterized):
        log.info(f"Downloading '{URL_UNCHARACTERIZED}'...")
        urllib.request.urlretrieve(URL_UNCHARACTERIZED, filename=fname_uncharacterized)
        log.info(f"Done downloading '{URL_UNCHARACTERIZED}'...")

    # Split
    split_path = os.path.join(target_dir, "splits.json")
    if split_file:
        log.info(f"Loading the provided split file: '{split_file}'.")
        splits = read_json(json_path=split_file)
        splits = {split: list(sorted(splits[split])) for split in splits}

    else:
        log.info(f"Creating the splits.")
        log.info(f"Reading the uncharacterized molecules...")
        uncharacterized_indices = read_uncharacterized(fname_uncharacterized)
        assert len(
            uncharacterized_indices) == N_UNCHARACTERIZED, f"Expected {N_UNCHARACTERIZED} uncharacterized molecules, but found {len(uncharacterized_indices)}."

        indices = np.array(sorted(list(set(range(N_GDB9)) - set(uncharacterized_indices))))
        n_indices = len(indices)
        assert n_indices == (
                N_GDB9 - N_UNCHARACTERIZED), f"Expected {N_GDB9 - N_UNCHARACTERIZED} molecules in total, but found {n_indices}."

        n_train = 100000
        n_test = int(0.1 * n_indices)
        n_val = n_indices - (n_train + n_test)

        np.random.seed(SEED)
        data_perm = np.random.permutation(n_indices)
        train_ptr, val_ptr, test_ptr, extra_ptr = np.split(
            data_perm, [n_train, n_train + n_val, n_train + n_val + n_test])
        assert (len(extra_ptr) == 0), 'Split was inexact {} {} {} {}'.format(
            len(train_ptr), len(val_ptr), len(test_ptr), len(extra_ptr))

        splits = {"train": list(sorted(indices[train_ptr].tolist())),
                  "val": list(sorted(indices[val_ptr].tolist())),
                  "test": list(sorted(indices[test_ptr].tolist()))}
        log.info(f"Done creating the splits.")

    log.info(f"Saving the splits for later reuse to: '{split_path}'.")
    save_json(json_dict=splits, json_path=split_path)

    # Create the preprocessed datasets
    preprocessed_dir = os.path.join(target_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    with tarfile.open(fname_gdb9, "r") as tf:
        files = tf.getmembers()
        for split in splits:
            log.info(f"Processing split: '{split}'.")

            split_indices = splits[split]
            split_metrics = QM9Metrics(summarize_hidden=True,
                                       hidden_prefix="")

            split_list = []

            for index in tqdm.tqdm(split_indices, desc=f"Processing split '{split}'", unit="mol"):
                file = tf.extractfile(files[index])
                atoms = read_file(file)

                data = process_atoms(atoms)
                split_metrics([atoms])

                split_list.append(data)

            log.info(f"Summarizing infos...")
            infos_path = os.path.join(preprocessed_dir, f"{split}_infos.json")
            agg_infos = split_metrics.summarize()
            save_json(agg_infos, infos_path)

            pt_path = os.path.join(preprocessed_dir, f"{split}.pt")
            torch.save(split_list, pt_path)
            log.info(f"Done processing split: '{split}'. Saved in '{pt_path}'.")


if __name__ == '__main__':
    fire.Fire(preprocess_qm9)
