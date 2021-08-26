# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:52:13 2021

@author: josep
"""

######### DATA PREPROCESSING AND MANIPULATION #########

import sys
from ase.io import read

# Let's use ASE to create atomic structures as ase.Atoms objects.

structures = []

name = str(sys.argv[1])
descriptor = str(sys.argv[2])
print("Loading data from file: {}".format(name))

structures = []
structures = read(name, index=":")

print("Number of systems: {}".format(len(structures)))

species = set()
for structure in structures:
    species.update(structure.get_chemical_symbols())
    
#collecting energies (output to predict)
print("Sorting ML labels...")

energies = []
for structure in structures:
    energies.append(float(structure.info["U0"]))

import numpy as np
print("Ordering data...")
#order structures in terms of increasing energy (U0)
indices = np.argsort(energies)
sorted_structures = []
for i in range(len(structures)):
    sorted_structures.append(structures[indices[i]])

sorted_energies = []
for structure in sorted_structures:
    sorted_energies.append(float(structure.info["U0"]))
    
#finding max value of atoms per molecule in structures
size = []
for structure in sorted_structures:
    size.append(len(structure))
    
max_atoms = np.max(size)
min_atoms = np.min(size)
print("Largest molecule has ", max_atoms, " atoms.")
print("Smallest molecule has ", min_atoms, " atoms.")

def get_CM_fv():
    
    #creating descriptor
    from dscribe.descriptors import CoulombMatrix
    
    cm = CoulombMatrix(n_atoms_max=max_atoms, flatten=True, sparse=False)
        
    feature_vector = cm.create(sorted_structures, n_jobs=8)
    
    return feature_vector

def get_SOAP_fv():
    #creating descriptor
    from dscribe.descriptors import SOAP

    soap = SOAP(
                 species=species,
                 periodic=False,
                 rcut=5,
                 nmax=4,
                 lmax=4,
                 average='outer',
                 sparse=False
                )
    feature_vector = soap.create(sorted_structures, n_jobs=8)
    return feature_vector

def get_ACSF_fv():
    
    #creating descriptor
    from dscribe.descriptors import ACSF
    
    acsf= ACSF(
        species=species,
        rcut=6.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )
    num_features = acsf.get_number_of_features()
    print("Number of features: " + str(num_features))

    feature_vector = acsf.create(sorted_structures, n_jobs=8, verbose=1)
    
    fv_holder = np.zeros((len(sorted_structures), max_atoms, num_features))
    
    for i in range(len(feature_vector)): # molecule
        for j in range(len(feature_vector[i])): # j atoms in the ith molecule
            print("Atom " + str(j) + " in Molecule " + str(i) + " with " + str(len(feature_vector[i])) + " features.")
            fv_holder[i][j] = np.array(feature_vector[i][j])

    feature_vector = np.reshape(fv_holder,(fv_holder.shape[0],fv_holder.shape[1]*fv_holder.shape[2]))
    return feature_vector
    
def get_MBTR_fv():
    
    #creating descriptor
    from dscribe.descriptors import MBTR
    
    atomic_numbers = []
    for structure in sorted_structures:
        atomic_numbers.append(np.max(structure.get_atomic_numbers()))
    max_atomic_number = np.max(atomic_numbers)
    
    mbtr = MBTR(
        species=species,
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 0, "max": max_atomic_number, "n": 100, "sigma": 0.01},
        },
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.01},
            "weighting": {"function": "exp", "scale": 0.05, "cutoff": 0.01},
        },
        k3={
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.01},
            "weighting": {"function": "exp", "scale": 0.05, "cutoff": 0.01},
        },
        periodic=False,
        normalization="l2_each",
 	flatten=True,
    ) 
    feature_vector = mbtr.create(sorted_structures, n_jobs=8)
    return feature_vector

print("Creating descriptors...")

if descriptor=="CM":
    feature_vector = get_CM_fv()
    
elif descriptor=="SOAP":
    feature_vector = get_SOAP_fv()
    
elif descriptor=="ACSF":
    feature_vector = get_ACSF_fv()

elif descriptor=="MBTR":
    feature_vector = get_MBTR_fv()
    
else:
    print("Invalid descriptor name! Select either 'CM', 'SOAP', 'ACSF' or 'MBTR'.")
    exit()

#selecting every fifth element from dataset for test set
#putting remaining in x and y
print("Setting up training and testing sets...")

x_test = []
y_test = []
x = []
y = []
for i in range(len(sorted_structures)):
    if i%5 == 0:
        x_test.append(feature_vector[i])
        y_test.append(sorted_energies[i])
    else:
        x.append(feature_vector[i])
        y.append(sorted_energies[i])
        
#shuffle x and y test sets randomly but both in same way
import random                           
temp = list(zip(x_test, y_test))
random.shuffle(temp)
x_test, y_test = zip(*temp)

#randomly split train and validation sets into 80% and 20% respectively
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2)

print("Saving datasets...")
#save datasets for consistency/reusability
np.save(str(descriptor) + "_x_train", x_train)
np.save(str(descriptor) + "_x_test", x_test)
np.save(str(descriptor) + "_x_val", x_val)
np.save(str(descriptor) + "_y_train", y_train)
np.save(str(descriptor) + "_y_test", y_test)
np.save(str(descriptor) + "_y_val", y_val)
    
    
