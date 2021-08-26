# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 08:48:07 2021

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

from dscribe.descriptors import ACSF

# Setting up the ACSF descriptor
acsf = ACSF(
    species=["H", "O"],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

# Create ACSF output for the hydrogen atom at index 1
pos=[]
for i in range(len(structures)):
    pos.append([0])
acsf_water = acsf.create(structures, positions=pos)

acsf_water = np.reshape(acsf_water,(acsf_water.shape[0],acsf_water.shape[2]))

print("Number of values for ACSF descriptor: {}".format(len(acsf_water[0])))


print("ACSF vector for first water molecule:")
print(acsf_water[0])
print(acsf_water.shape)
print(type(acsf_water[0]))
print("Output energy value for system:")
print(values[0])

#x = np.array(mbtr_water) #~ 200 MSE
x = np.array(acsf_water) #~1 MSE
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

#EARLY STOPPING Callback
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True,min_delta=0.01)
cb_list = [es]

model = generate_NN(10,100, "relu", 1)
history = model.fit(x_train, y_train, epochs=100, validation_split=1/5, verbose=1)
_,test_mse = model.evaluate(x_test,y_test)
print("Model has Mean Squared Error: ", test_mse)


tf.keras.backend.clear_session()
    
