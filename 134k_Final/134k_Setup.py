# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:52:13 2021

@author: josep
"""

######### DATA PREPROCESSING AND MANIPULATION #########
import time
import sys
from ase.io import read

# Let's use ASE to create atomic structures as ase.Atoms objects.

structures_train = []
structures_val = []
structures_test = []

name_train = "train_set.xyz"
name_val = "val_set.xyz"
name_test = "test_set.xyz"

descriptor = str(sys.argv[1])
cores = int(sys.argv[2])
print("Requesting " + str(cores) + " core(s)...")
print("Loading training data from file: {}".format(name_train))
print("Loading validation data from file: {}".format(name_val))
print("Loading testing data from file: {}".format(name_test))

structures_train = []
structures_train = read(name_train, index=":")

structures_val = []
structures_val = read(name_val, index=":")

structures_test = []
structures_test = read(name_test, index=":")

print("Number of systems in training set: {}".format(len(structures_train)))
print("Number of systems in validation set: {}".format(len(structures_val)))
print("Number of systems in testing set: {}".format(len(structures_test)))

species = set()
for structure in structures_train:
    species.update(structure.get_chemical_symbols())
for structure in structures_val:
    species.update(structure.get_chemical_symbols())
for structure in structures_test:
    species.update(structure.get_chemical_symbols())
    
#collecting energies (output to predict)
print("Sorting ML labels...")
y_train = []
for structure in structures_train:
    y_train.append(float(structure.info["U0"]))
y_val = []
for structure in structures_val:
    y_val.append(float(structure.info["U0"]))
y_test = []
for structure in structures_test:
    y_test.append(float(structure.info["U0"]))

import numpy as np
    
size = []
for structure in structures_train:
    size.append(len(structure))
for structure in structures_val:
    size.append(len(structure))
for structure in structures_test:
    size.append(len(structure))    
max_atoms = np.max(size)
min_atoms = np.min(size)
print("Largest molecule has ", max_atoms, " atoms.")
print("Smallest molecule has ", min_atoms, " atoms.")

def get_CM_fv(structures, name):
    
    #creating descriptor
    from dscribe.descriptors import CoulombMatrix
    
    cm = CoulombMatrix(n_atoms_max=max_atoms)
    start = time.time()    
    feature_vector = cm.create(structures, n_jobs=cores)
    end = time.time()
    print("Time to create CM descriptors for " + name + ": {}".format(end-start))
    return feature_vector

def get_SOAP_fv(structures, name):
    #creating descriptor
    from dscribe.descriptors import SOAP

    soap = SOAP(
                 species=species,
                 periodic=False,
                 rcut=6.0,
                 nmax=8,
                 lmax=6,
                )
    start = time.time()
    feature_vector = soap.create(structures, n_jobs=cores)
    end = time.time()
    print("Time to create SOAP descriptors for " + name + ": {}".format(end-start))
    return feature_vector

def get_ACSF_fv(structures, name):
    
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
    start = time.time()
    feature_vector = acsf.create(structures, n_jobs=cores)
    end = time.time()
    print("Time to create ACSF descriptors for " + name + ": {}".format(end-start))
    fv_holder = np.zeros((len(structures), max_atoms, num_features))
    
    for i in range(len(feature_vector)): # molecule
        for j in range(len(feature_vector[i])): # j atoms in the ith molecule
            print("Atom " + str(j) + " in Molecule " + str(i) + " with " + str(len(feature_vector[i])) + " features.")
            fv_holder[i][j] = np.array(feature_vector[i][j])

    feature_vector = np.reshape(fv_holder,(fv_holder.shape[0],fv_holder.shape[1]*fv_holder.shape[2]))
    return feature_vector
    
def get_MBTR_fv(structures, name):
    
    #creating descriptor
    from dscribe.descriptors import MBTR
    
    atomic_numbers = []
    for structure in structures:
        atomic_numbers.append(np.max(structure.get_atomic_numbers()))
    max_atomic_number = np.max(atomic_numbers)
    
    mbtr = MBTR(
        species=species,
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 0, "max": max_atomic_number, "n": 100, "sigma": 0.1},
        },
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3},
        },
        k3={
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3},
        },
        periodic=False,
        normalization="l2_each",
    )
    start = time.time() 
    feature_vector = mbtr.create(structures, n_jobs=cores)
    end = time.time()
    print("Time to create MBTR descriptors for " + name + ": {}".format(end-start))
    return feature_vector

print("Creating descriptors...")

if descriptor=="CM":
    x_train = get_CM_fv(structures_train,"training set")
    x_test = get_CM_fv(structures_test, "testing set")
    x_val = get_CM_fv(structures_val, "validation set")
elif descriptor=="SOAP":
    x_train = get_SOAP_fv(structures_train, "training set")
    x_test = get_SOAP_fv(structures_test, "testing set")
    x_val = get_SOAP_fv(structures_val, "validation set")
elif descriptor=="ACSF":
    x_train = get_ACSF_fv(structures_train, "training set")
    x_test = get_ACSF_fv(structures_test, "testing set")
    x_val = get_ACSF_fv(structures_val, "validation set")
elif descriptor=="MBTR":
    x_train  = get_MBTR_fv(structures_train, "training set")
    x_test = get_MBTR_fv(structures_test, "testing set")
    x_val = get_MBTR_fv(structures_val, "validation set")
else:
    print("Invalid descriptor name! Select either 'CM', 'SOAP', 'ACSF' or 'MBTR'.")
    exit()

print("Saving datasets...")
#save datasets for consistency/reusability
np.save(str(descriptor) + "_x_train", x_train)
np.save(str(descriptor) + "_x_test", x_test)
np.save(str(descriptor) + "_x_val", x_val)
np.save(str(descriptor) + "_y_train", y_train)
np.save(str(descriptor) + "_y_test", y_test)
np.save(str(descriptor) + "_y_val", y_val)
