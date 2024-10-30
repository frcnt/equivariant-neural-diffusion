import json
from typing import Sequence

import ase
import ase.io
import numpy as np
import torch


def read_json(json_path: str):
    with open(json_path, encoding="utf-8", mode="r") as fp:
        return json.load(fp)


def save_json(json_dict: dict, json_path: str):
    def _fix_dict():
        for key in json_dict:
            if isinstance(json_dict[key], np.ndarray):
                json_dict[key] = json_dict[key].tolist()

    _fix_dict()
    with open(json_path, encoding="utf-8", mode="w") as fp:
        json.dump(json_dict, fp)


def save_images(images: Sequence[ase.Atoms], filename: str):
    ase.io.write(filename=filename, images=images, format="xyz")


def atoms_from_tensors(h: torch.Tensor,
                       pos: torch.Tensor,
                       ptr: torch.Tensor,
                       decoder: list[str]) -> list[ase.Atoms]:
    h = h.to("cpu")
    pos = pos.to("cpu")
    ptr = ptr.to("cpu")

    if h.ndim > 1 and h.shape[1] > 1:
        h = torch.argmax(h, dim=1)

    atoms_list = []

    for start_idx, end_idx in zip(ptr[:-1], ptr[1:]):
        positions = pos[start_idx:end_idx, :]
        symbols = [decoder[idx.item()] for idx in h[start_idx:end_idx]]
        atoms = ase.Atoms(symbols=symbols, positions=positions)
        atoms_list.append(atoms)
    return atoms_list
