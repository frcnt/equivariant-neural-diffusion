"""
Adapted from EEGSDE.
"""
import ase
import numpy as np


def calc_fp_ob(atoms: ase.Atoms, center: bool = True, order: bool = True):
    try:
        from openbabel import pybel, openbabel as ob
    except:
        raise ImportError("Could not import openbabel")

    if center:
        atoms.positions -= atoms.positions.mean(axis=0)
    pos = atoms.positions
    numbers = atoms.numbers

    # order atoms by distance to center of mass
    if order:
        d = np.sum(pos ** 2, axis=1)
        sorted_idx = np.argsort(d)
        pos = pos[sorted_idx]
        numbers = numbers[sorted_idx]

    # Open Babel OBMol representation
    obmol = ob.OBMol()
    obmol.BeginModify()
    # set positions and atomic numbers of all atoms in the molecule
    for p, n in zip(pos, numbers):
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(int(n))
        obatom.SetVector(*p.tolist())
    # infer bonds and bond order
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    obmol.EndModify()
    _fp = pybel.Molecule(obmol).calcfp()
    fp_bits = {*_fp.bits}

    fp_32 = np.array(_fp.fp, dtype=np.uint32)
    # convert fp to 1024bit
    fp_1024 = np.array(fp_32, dtype='<u4')
    fp_1024 = np.unpackbits(fp_1024.view(np.uint8), bitorder='little')

    return fp_bits, fp_1024
