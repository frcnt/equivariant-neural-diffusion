import ase
import numpy as np
from rdkit import Chem

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf

BONDS1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}

BONDS3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

STDV = {"H": 5, "C": 1, "N": 1, "O": 2, "F": 3}
MARGIN1, MARGIN2, MARGIN3 = 10, 5, 3

ALLOWED_BONDS = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}

BOND_LIST = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]


def get_bond_order(atom1, atom2, distance, check_exists=True, single_bond=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in BONDS1:
            print(f"Atom {atom1} not in bonds1")
            return 0
        if atom2 not in BONDS1[atom1]:
            print(f"Atom {atom2} not in bonds1[{atom1}]")
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < BONDS1[atom1][atom2] + MARGIN1:
        # Check if atoms in bonds2 dictionary.
        if atom1 in BONDS2 and atom2 in BONDS2[atom1]:
            thr_bond2 = BONDS2[atom1][atom2] + MARGIN2
            if distance < thr_bond2:
                if atom1 in BONDS3 and atom2 in BONDS3[atom1]:
                    thr_bond3 = BONDS3[atom1][atom2] + MARGIN3
                    if distance < thr_bond3:
                        return 3 if not single_bond else 1  # Triple
                return 2 if not single_bond else 1  # Double
        return 1  # Single
    return 0  # No bond


def check_stability(
        atoms: ase.Atoms,
        bond_order_per_atom: np.ndarray,
):
    num_atoms = len(atoms)
    atom_stable = 0
    for symbol_i, nr_bonds_i in zip(atoms.symbols, bond_order_per_atom):
        possible_bonds = ALLOWED_BONDS[symbol_i]
        if type(possible_bonds) == int:
            is_stable = (possible_bonds == nr_bonds_i)
        else:
            is_stable = nr_bonds_i in possible_bonds
        atom_stable += int(is_stable)

    molecule_stable = (atom_stable == num_atoms)
    return molecule_stable, atom_stable, num_atoms


def _print_table(bonds_dict):
    letters = ["H", "C", "O", "N", "P", "S", "F", "Si", "Cl", "Br", "I"]

    new_letters = []
    for key in letters + list(bonds_dict.keys()):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f"{x} & ", end="")
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f"{y} & ", end="")
            if x in bonds_dict[y]:
                print(f"{bonds_dict[y][x]} & ", end="")
            else:
                print("- & ", end="")
        print()


# print_table(bonds3)


def _check_consistency_bond_dictionaries():
    for bonds_dict in [BONDS1, BONDS2, BONDS3]:
        for atom1 in BONDS1:
            if atom1 in bonds_dict:
                for atom2 in bonds_dict[atom1]:
                    bond = bonds_dict[atom1][atom2]
                    try:
                        bond_check = bonds_dict[atom2][atom1]
                    except KeyError:
                        raise ValueError("Not in dict " + str((atom1, atom2)))

                    assert (
                            bond == bond_check
                    ), f"{bond} != {bond_check} for {atom1}, {atom2}"


if __name__ == '__main__':
    _print_table(BONDS1)
    print()
    _print_table(BONDS2)
    print()
    _print_table(BONDS3)
