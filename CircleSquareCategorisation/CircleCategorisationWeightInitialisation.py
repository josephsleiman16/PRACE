# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:02:30 2021

@author: josep
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import he_normal, he_uniform, glorot_normal, glorot_uniform, random_normal, random_uniform

Ndatapoints = 10000

def load_data(datafile):
    dataset = np.zeros((Ndatapoints,3))
    f = open(datafile,"r")
    lines=f.readlines()[1:]
    datapoint = -1
    counter = 0
    for x in lines:
        if x == '\n':
            datapoint +=1
            continue
        elif counter == 0:
            dataset[datapoint][counter] = float(x)
            counter = 1
            continue
        elif counter == 1:
            dataset[datapoint][counter] = float(x)
            counter = 2
            continue
        elif counter == 2:
            dataset[datapoint][counter] = int(x)
            counter = 0
            continue
    f.close()
    return dataset

def split(data, testsize):
    df = pd.DataFrame(data)
    x = df.drop(columns=2)
    y = df[2]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=testsize, train_size=1-testsize, random_state=1)
#     x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=18/20, random_state=1)
    return x_train, y_train, x_test, y_test

def generate_NN(n_hidden_layers, hidden_neurons, hidden_activation, output_neurons, output_activation, opt, initializer):
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.Flatten()) #flattens dimensions of data so its 1D and takes as input
    model.add(tf.keras.layers.Dense(hidden_neurons,activation=hidden_activation,input_dim=2, kernel_initializer=initializer))
    for i in range(n_hidden_layers-1): #n_hidden_layers hidden layers with hidden_neurons nerons using the hidden_activation 
        model.add(tf.keras.layers.Dense(hidden_neurons, activation=hidden_activation))
    model.add(tf.keras.layers.Dense(output_neurons,activation=output_activation)) #final output layer
    
    #loss function compares y_pred to y_true
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

#EARLY STOPPING Callback
#es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True,min_delta=0.01)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True,min_delta=0.01)

cb_list = [es]

data = load_data("circle_square.data")
x_train, y_train, x_test, y_test = split(data, 0.18)

inits=[he_uniform(), he_normal(), glorot_uniform(), glorot_normal(), random_uniform(), random_normal()]
init_names=["HeUniform", "HeNormal", "GlorotUniform", "GlorotNormal", "RandomUniform", "RandomNormal"]
for i in range(len(inits)):               
    opt = RMSprop()
    model2 = generate_NN(3, 128, "relu", 1, "sigmoid", opt, inits[i])
    history = model2.fit(x_train.values, y_train.values, epochs=20, validation_split=1/41, callbacks=cb_list, verbose=0)
    _,test_acc = model2.evaluate(x_test,y_test)
    print("Model with Weight Initialiser : ", init_names[i], ", has Test Accuracy: ", test_acc)
    plt.plot(history.history['acc'],label="train")
    plt.plot(history.history['val_acc'],label="val")
    plt.title("Weight Initialiser = " + str(init_names[i]))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("AccuracyFor"+str(init_names[i])+".png")
    plt.clf()
    
tf.keras.backend.clear_session()
