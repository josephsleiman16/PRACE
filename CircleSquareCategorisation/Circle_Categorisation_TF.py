#!/usr/bin/env python
# coding: utf-8

# In[246]:


import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np


# In[247]:


Ndatapoints = 10000


# In[248]:


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


# In[249]:


def split(data, testsize):
    df = pd.DataFrame(data)
    x = df.drop(columns=2)
    y = df[2]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=testsize, train_size=1-testsize, random_state=1)
#     x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=18/20, random_state=1)
    return x_train, y_train, x_test, y_test


# In[292]:


def generate_NN(n_hidden_layers, hidden_neurons, hidden_activation, output_neurons, output_activation, opt):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) #flattens dimensions of data so its 1D and takes as input
    for i in range(n_hidden_layers): #n_hidden_layers hidden layers with hidden_neurons nerons using the hidden_activation 
        model.add(tf.keras.layers.Dense(hidden_neurons, activation=hidden_activation))
    model.add(tf.keras.layers.Dense(output_neurons,activation=output_activation)) #final output layer
    
    #loss function compares y_pred to y_true
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[338]:


#EARLY STOPPING Callback
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, restore_best_weights=True,min_delta=0.01)
cb_list = [es]


# In[289]:


data = load_data("circle_square.data")
x_train, y_train, x_test, y_test = split(data, 0.18)


# In[342]:


model = generate_NN(1,128,"sigmoid",1,"sigmoid","adam") # 1 hidden layer with 128 neurons (relu), 1 neuron in output layer (sigmoid)
model.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model.evaluate(x_test,y_test)


# In[315]:


model1 = generate_NN(1,128,"relu",1,"sigmoid","adam") # 1 hidden layer with 128 neurons (relu), 1 neuron in output layer (sigmoid)
model1.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)


# In[316]:


model1.evaluate(x_test, y_test)


# In[317]:


predictions = model1.predict(x_test)


# In[318]:


val = 0
print(predictions[val])
print(y_test.values[val])


# In[319]:


model2 = generate_NN(1, 128, "relu", 2, "sigmoid", "adam")# 1 hidden layer with 128 neurons (relu), 2 output neuron in output layer (sigmoid)


# In[320]:


model2.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)


# In[321]:


predictions = model2.predict(x_test)


# In[322]:


val = 7
print(predictions[val])
print(y_test.values[val])


# In[334]:


model3a = generate_NN(3, 128, "relu", 2, "sigmoid", "adam") #BEST MODEL
model3a.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)


# In[335]:


model3b = generate_NN(4, 128, "relu", 2, "sigmoid", "adam") 
model3b.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)


# In[336]:


model3a.evaluate(x_test,y_test) #BEST MODEL


# In[337]:


model3b.evaluate(x_test,y_test)


# In[340]:


model4 = generate_NN(3,128,"relu",2,"sigmoid","adamax")
model4.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[329]:


model4 = generate_NN(3,128,"relu",2,"sigmoid","SGD")
model4.fit(x_train.values,y_train.values,epochs=50, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[330]:


model4 = generate_NN(3,128,"relu",2,"sigmoid","adagrad")
model4.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[331]:


opt = tf.keras.optimizers.Ftrl(learning_rate=0.001)
model4 = generate_NN(3,128,"relu",2,"sigmoid",opt)
model4.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[339]:


model4 = generate_NN(3,128,"relu",2,"sigmoid","adadelta")
model4.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[332]:


opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model4 = generate_NN(3,128,"relu",2,"sigmoid",opt)
model4.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[333]:


opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model4 = generate_NN(3,128,"relu",2,"sigmoid",opt)
model4.fit(x_train.values,y_train.values,epochs=30, validation_split=1/41,callbacks=cb_list)
model4.evaluate(x_test,y_test)


# In[ ]:




