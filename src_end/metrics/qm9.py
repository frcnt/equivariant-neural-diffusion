from pathlib import Path
from typing import Optional

import ase
import numpy as np

from src_end.data.utils import read_json
from src_end.metrics.base import Metrics, discrete_histogram, ContextMetrics
from src_end.metrics.bonds import check_stability
from src_end.metrics.fp import calc_fp_ob
from src_end.metrics.rdkit_utils import make_mol_rdkit_qm9, check_validity

_SYMBOLS_QM9 = ["H", "C", "N", "O", "F"]


class QM9Metrics(Metrics):
    def __init__(self,
                 atom_types_str: str = _SYMBOLS_QM9,
                 max_num_atoms: int = 29,
                 json_path: Optional[str | Path] = None,
                 summarize_hidden: bool = False,
                 hidden_prefix: str = "_"):

        self.encoder = {s: idx for idx, s in enumerate(atom_types_str)}
        self.max_num_atoms = max_num_atoms

        if json_path:
            dataset_infos = read_json(json_path=json_path)
            ref_smiles = set(dataset_infos.get("smiles"))
            ref_atom_hist = np.array(dataset_infos.get("atom_hist"))
        else:
            ref_smiles = set([])
            ref_atom_hist = None

        self.ref_smiles = ref_smiles
        self.ref_atom_hist = ref_atom_hist

        self.summarize_hidden = summarize_hidden
        self.hidden_prefix = hidden_prefix

        self.smiles = ...
        self.valid = ...
        self.valid_connected = ...
        self.n_atoms = ...
        self.molecule_stable = ...
        self.atom_hist = ...
        self.atom_stable = ...

        self.reset()

    def update(self, atoms: list[ase.Atoms] | ase.Atoms):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]

        for a in atoms:
            raw_mol, bond_order_per_atom = make_mol_rdkit_qm9(a)
            mol, smi, v, c = check_validity(raw_mol)
            molecule_stable, atom_stable, n_atoms = check_stability(a, bond_order_per_atom)

            self.smiles.append(smi)
            self.valid.append(v)
            self.valid_connected.append(c)
            self.n_atoms.append(n_atoms)
            self.molecule_stable.append(molecule_stable)
            self.atom_stable.append(atom_stable)
            self.atom_hist.append(discrete_histogram(a.symbols, encoder=self.encoder))

    def summarize(self) -> dict:
        assert len(self.valid) == len(self.valid_connected)
        assert len(self.valid) == len(self.molecule_stable)
        assert len(self.valid) == len(self.smiles)

        n_samples = len(self.valid)
        n_atoms = sum(self.n_atoms)

        summary = {}

        summary["atom_stable"] = sum(self.atom_stable) / n_atoms
        summary["molecule_stable"] = sum(self.molecule_stable) / n_samples

        summary["valid"] = sum(self.valid) / n_samples
        summary["valid_connected"] = sum(self.valid_connected) / n_samples

        valid_unique_smiles = set([smiles for (v, smiles) in zip(self.valid, self.smiles) if v])
        summary["valid_unique"] = len(valid_unique_smiles) / n_samples
        if self.ref_smiles is not None:
            vun_smiles = valid_unique_smiles.difference(self.ref_smiles)
            summary["valid_unique_novel"] = len(vun_smiles) / n_samples

        atom_hist = np.sum(np.stack(self.atom_hist, axis=0), axis=0)
        atom_hist = (atom_hist / atom_hist.sum())
        if self.ref_atom_hist is not None:
            summary["tv_atom"] = np.sum(np.abs(self.ref_atom_hist - atom_hist)).item()

        if self.summarize_hidden:
            summary[f"{self.hidden_prefix}atom_hist"] = atom_hist
            summary[f"{self.hidden_prefix}num_atoms_hist"] = discrete_histogram(self.n_atoms,
                                                                                encoder={idx: idx for idx in
                                                                                         range(self.max_num_atoms + 1)},
                                                                                norm=True)
            summary[f"{self.hidden_prefix}smiles"] = list(valid_unique_smiles)

        return summary

    def reset(self):
        self.smiles = []
        self.valid = []
        self.valid_connected = []
        self.n_atoms = []
        self.molecule_stable = []
        self.atom_hist = []
        self.atom_stable = []


class OBFPMetrics(ContextMetrics):
    def __init__(self):
        self.similarity = ...
        self.reset()

    def update(self, atoms: list[ase.Atoms] | ase.Atoms, targets: np.ndarray):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]

        for idx, (sample, target_fp) in enumerate(zip(atoms, targets)):
            _, sample_fp = calc_fp_ob(sample)
            intersection = np.sum(sample_fp * target_fp)
            union = np.sum(sample_fp) + np.sum(target_fp) - intersection
            if union > 0.0:
                sim = intersection / union
            else:
                sim = 1.0
            self.similarity.append(sim)

    def summarize(self) -> dict:
        summary = {"fp_similarity": np.mean(self.similarity)}
        return summary

    def reset(self):
        self.similarity = []


class CompositionMetrics(ContextMetrics):
    def __init__(self):
        self.matching = ...
        self.reset()

    def update(self, atoms: list[ase.Atoms] | ase.Atoms, targets: list[str]):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]

        for idx, (sample, target_composition) in enumerate(zip(atoms, targets)):
            m = sample.get_chemical_formula() == target_composition
            self.matching.append(m)

    def summarize(self) -> dict:
        summary = {"matching_composition": np.mean(self.matching)}
        return summary

    def reset(self):
        self.matching = []
