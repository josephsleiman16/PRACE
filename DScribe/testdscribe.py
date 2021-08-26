import numpy as np
import matplotlib.pyplot as plt
from ase.calculators.lj import LennardJones
from ase.build import molecule
from ase import Atoms
import dscribe
from dscribe.descriptors import SOAP
import sklearn
import tensorflow

def main():
    structure1 = molecule("H2O")
    structure2 = Atoms(symbols=["C","O"], positions=[[0,0,0], [1.128,0,0]])
    structures = [structure1, structure2]

    species = set()
    for structure in structures:
        species.update(structure.get_chemical_symbols())

    soap = SOAP(
            species=species,
            periodic=False,
            rcut=5,
            nmax=8,
            lmax=8,
            average="outer",
            sparse=False
    )

    feature_vectors = soap.create(structures)
    print(feature_vectors.shape)
    print(len(feature_vectors))
    print(feature_vectors)
    return


main()

