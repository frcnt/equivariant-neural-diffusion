from functools import partial

import ase
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from src_end.metrics.bonds import BOND_LIST, get_bond_order

__all__ = ["make_mol_rdkit_qm9", "check_validity"]


def _make_mol_rdkit(
        atoms: ase.Atoms,
        single_bond: bool,
        with_conformer: bool = False
):
    distances = atoms.get_all_distances()
    num_atoms = len(atoms)
    bond_order_per_atom = np.zeros(num_atoms, dtype="int")

    mol = Chem.RWMol()

    # Add all atoms first
    for symbol in atoms.symbols:
        a = Chem.Atom(symbol)
        mol.AddAtom(a)

    # Add all bonds
    for i in range(num_atoms):

        symbol_i = atoms.symbols[i]

        for j in range(i + 1, num_atoms):
            dist = distances[i, j]
            symbol_j = atoms.symbols[j]
            order = get_bond_order(
                symbol_i, symbol_j, dist, single_bond=single_bond
            )

            bond_order_per_atom[i] += order
            bond_order_per_atom[j] += order

            if order > 0:
                mol.AddBond(i, j, BOND_LIST[order])

    if with_conformer:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x, y, z = atoms.positions[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf)

    return mol, bond_order_per_atom


make_mol_rdkit_qm9 = partial(_make_mol_rdkit, single_bond=False)


def check_validity(mol: Chem.Mol):
    def mol_to_smi(mol_: Chem.Mol):
        try:
            Chem.SanitizeMol(mol_)
            smi = Chem.MolToSmiles(mol_, canonical=True)
            return mol_, smi
        except ValueError:
            return None, None

    mol, smi = mol_to_smi(mol)
    v, c = 0, 0

    if smi is not None:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        c = int(len(mol_frags) == 1)

        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        mol, smi = mol_to_smi(largest_mol)
        v = int(smi is not None)

    return mol, smi, v, c
