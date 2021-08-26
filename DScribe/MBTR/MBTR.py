# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 08:46:14 2021

@author: josep
"""

 
################### dscribe part #################

from ase.io import read
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np

# Let's use ASE to create atomic structures as ase.Atoms objects.
print("Loading data...")

name= 'water_all.xyz'

structures = read(name, index=":")

print("Number of systems: {}".format(len(structures)))

for system in structures:
    print("Energy: {}".format(system.info["energy"]))
    print("Positions of atoms: {}".format(system.get_positions()))
    print("Species of atoms: {}".format(system.get_chemical_symbols()))
#    print("Charge of atoms: {}".format(system.get_initial_charges()))

values = []

for system in structures:
    values.append(float(system.info["energy"]))


#print(values)

# Let's create a list of structures and gather the chemical elements that are
# in all the structures.

print("Creating descriptors...")

species = set()
for structure in structures:
    species.update(structure.get_chemical_symbols())

#print(species)


# Let's configure the descriptor.
from dscribe.descriptors import MBTR


mbtr = MBTR(
    species=["H", "O"],
    k1={
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 8, "n": 50, "sigma": 0.1},
    },
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1, "n": 50, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
    },
    k3={
        "geometry": {"function": "cosine"},
        "grid": {"min": -1, "max": 1, "n": 50, "sigma": 0.1},
        "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
    },
    periodic=False,
    normalization="l2_each", 
    flatten = True
)

# Let's create MBTR feature vectors for each structure
mbtr_water = mbtr.create(structures, n_jobs=1)

print("Number of values for MBTR descriptor: {}".format(len(mbtr_water[0])))

print("MBTR vector for first water molecule:")
print(mbtr_water[0])
print(mbtr_water.shape)
print("Output energy value for system:")
print(values[0])

#x = np.array(mbtr_water) #~ 200 MSE
x = np.array(mbtr_water) #~1 MSE
# x = np.array(soap_water) #~5 MSE
y = np.array(values)
inputdim=x.shape[1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

def generate_NN(n_hidden_layers, hidden_neurons, hidden_activation, output_neurons):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(hidden_neurons,activation=hidden_activation,input_dim=inputdim, kernel_initializer="he_normal"))
    for i in range(n_hidden_layers-1): #n_hidden_layers hidden layers with hidden_neurons nerons using the hidden_activation 
        model.add(tf.keras.layers.Dense(hidden_neurons, activation=hidden_activation, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dense(output_neurons, kernel_initializer="he_normal")) #final output layer (no activation function since regression)
    
    #loss function compares y_pred to y_true
    model.compile(optimizer="adam",loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


model = generate_NN(10,100, "relu", 1)
history = model.fit(x_train, y_train, epochs=100, validation_split=1/5, verbose=1)
_,test_mse = model.evaluate(x_test,y_test)
print("Model has Mean Squared Error: ", test_mse)


tf.keras.backend.clear_session()
    
