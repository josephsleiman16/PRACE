# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:02:52 2021

@author: josep
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam

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

def generate_NN(n_hidden_layers, hidden_neurons, hidden_activation, output_neurons, output_activation, opt):
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.Flatten()) #flattens dimensions of data so its 1D and takes as input
    model.add(tf.keras.layers.Dense(hidden_neurons,activation=hidden_activation,input_dim=2))
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

#model = generate_NN(3, 128, "relu", 2, "sigmoid", "adam")
#model.fit(x_train.values, y_train.values, epochs=100, validation_split=1/41, callbacks=cb_list,verbose=0)
#_,test_acc = model.evaluate(x_test,y_test)
#print("Model 1 Test Accuracy: ", test_acc)

#model1 = generate_NN(4, 128, "relu", 2, "sigmoid", "adam")
#model1.fit(x_train.values, y_train.values, epochs=100, validation_split=1/41, callbacks=cb_list, verbose=0)
#_,test_acc = model1.evaluate(x_test,y_test)
#print("Model 2 Test Accuracy: ", test_acc)

learningRate=[1,0.1,0.01,0.001,0.0001,0.00001]
#adam
for lr in learningRate:        
    opt = Adam(lr=lr)
    model2 = generate_NN(3, 128, "relu", 1, "sigmoid", opt=opt)
    history = model2.fit(x_train.values, y_train.values, epochs=20, validation_split=1/41, callbacks=cb_list, verbose=0)
    _,test_acc = model2.evaluate(x_test,y_test)
    print("Model with Adam Optimizer and Learning Rate: ", lr, ", has Test Accuracy: ", test_acc)
    plt.plot(history.history['acc'],label="train")
    plt.plot(history.history['val_acc'],label="val")
    plt.title("Adam with Learning Rate = " + str(lr))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("AccuracyForAdam"+str(lr)+".png")
    plt.clf()
   
#adagrad
for lr in learningRate:        
    opt = Adagrad(lr=lr)
    model2 = generate_NN(3, 128, "relu", 1, "sigmoid", opt=opt)
    history = model2.fit(x_train.values, y_train.values, epochs=20, validation_split=1/41, callbacks=cb_list, verbose=0)
    _,test_acc = model2.evaluate(x_test,y_test)
    print("Model with Adagrad Optimizer and Learning Rate: ", lr, ", has Test Accuracy: ", test_acc)
    plt.plot(history.history['acc'],label="train")
    plt.plot(history.history['val_acc'],label="val")
    plt.title("Adagrad with Learning Rate = " + str(lr))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("AccuracyForAdagrad"+str(lr)+".png")
    plt.clf()

#RMSprop
for lr in learningRate:        
    opt = RMSprop(lr=lr)
    model2 = generate_NN(3, 128, "relu", 1, "sigmoid", opt=opt)
    history = model2.fit(x_train.values, y_train.values, epochs=20, validation_split=1/41, callbacks=cb_list, verbose=0)
    _,test_acc = model2.evaluate(x_test,y_test)
    print("Model with RMSprop Optimizer and Learning Rate: ", lr, ", has Test Accuracy: ", test_acc)
    plt.plot(history.history['acc'],label="train")
    plt.plot(history.history['val_acc'],label="val")
    plt.title("RMSprop with Learning Rate = " + str(lr))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("AccuracyForRMSprop"+str(lr)+".png")
    plt.clf()
    
tf.keras.backend.clear_session()
