# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:24:19 2021

@author: josep
"""

import numpy as np
import tensorflow as tf
import sys

descriptor = str(sys.argv[1])
# if descriptor != "CM" or descriptor != "ACSF" or descriptor != "MBTR" or descriptor!= "SOAP":
#     print("Invalid descriptor name! Select either 'CM', 'SOAP', 'ACSF' or 'MBTR'.")
#     exit()
    
x_train = np.load(str(descriptor) + "_x_train.npy", allow_pickle=True)
y_train = np.load(str(descriptor) + "_y_train.npy", allow_pickle=True)
x_val = np.load(str(descriptor) + "_x_val.npy", allow_pickle=True)
y_val = np.load(str(descriptor) + "_y_val.npy", allow_pickle=True)
x_test = np.load(str(descriptor) + "_x_test.npy", allow_pickle=True)
y_test = np.load(str(descriptor) + "_y_test.npy", allow_pickle=True)

if descriptor == 'CM' or descriptor == 'SOAP':
    max_fv = len(x_train[0])

else:
    size = []
    for structure in x_train:
        print(len(structure))
        size.append(len(structure))
    for structure in x_test:
        size.append(len(structure))
        print(len(structure))

    for structure in x_val:
        size.append(len(structure))
        print(len(structure))

    max_fv = np.max(size)
print("Number of max descriptors: {}".format(max_fv))


# ################### NN part ################################

print("NN workflow...")

import os
# GPU OFF
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
# Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass

def setup_NN(model, size_input, NN_top):

    # add input layer (input_shape=(size_input,)) and first hidden layer to NN
    model.add(tf.keras.layers.Dense(
        NN_top[0],
        input_shape=(size_input,),
      #   kernel_initializer = tf.keras.initializers.RandomUniform(minval= -2., maxval= 2., seed=None),
        kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
        activation='softplus',
        use_bias=True))

    # add hidden layers to NN 
    for i in range(len(NN_top)-1):
        model.add(tf.keras.layers.Dense(
            NN_top[i+1],
            kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
            activation='softplus',
            use_bias=True))

    # add output layer (energy)
    model.add(tf.keras.layers.Dense(
        1,
        kernel_initializer = tf.keras.initializers.RandomUniform(seed=None),
        activation='linear',
        use_bias=True))

    # save and print NN details
    print(model.summary())

    #tf.keras.utils.plot_model(model, to_file='NN_graph.png')
    #model.save_weights("NN_weights")
    #print(model.get_weights())
    #print(model.to_json())


size_input = max_fv

# NN initialization 
NN = tf.keras.Sequential()

# NN setup
NN_top = []

NN_top.append(100)
NN_top.append(50)
NN_top.append(5)

setup_NN(NN, size_input, NN_top)

# NN build
n_epoch = 100

opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
NN.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
history = NN.fit(x_train, y_train, epochs = n_epoch, batch_size = 32, verbose = 0, validation_data=(x_val, y_val))


# PLOT AND STATS RESULTS
import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['mean_squared_error'], label="train")
plt.plot(history.history['val_mean_squared_error'],label="val")
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', "val"], loc='upper left')
plt.savefig('train_and_val_process.png')
plt.close()



# Linear regression model of results
from sklearn import linear_model
from scipy import stats

# Train data stat and plot
inputs = x_train
predicted = NN.predict(inputs)
expected = y_train

mse = NN.evaluate(inputs, expected)
mse = round(mse[0],3)

# sklearn linear regression
regr = linear_model.LinearRegression()
regr.fit(expected[:, np.newaxis], predicted)


# SciPy linear regression
predicted = np.reshape(predicted,-1)

slope, intercept, R, p_value, std_err = stats.linregress(expected, predicted)
R2 = R*R
R2= round(R2, 3)
slope = round(slope, 3)
intercept = round(intercept, 3)

if intercept < 0:
    lin_r = 'y = '+str(slope)+'x'+str(intercept)
else:
    lin_r = 'y = '+str(slope)+'x+'+str(intercept)


print('Train data analysis')
print(lin_r)
print('R^2: ', R2)
print('mse: ', mse)


plt.plot(expected, predicted, 'bo', expected, expected, 'r', expected, regr.predict(expected[:, np.newaxis]), 'g')
plt.xlabel('expected')
plt.ylabel('predicted')
plt.savefig('train.png')
plt.close()

# Test data stat and plot
inputs = x_test
predicted = NN.predict(inputs)
expected = y_test

mse = NN.evaluate(inputs, expected)
mse = round(mse[0],3)

# sklearn linear regression
regr = linear_model.LinearRegression()
regr.fit(expected[:, np.newaxis], predicted)


# SciPy linear regression
predicted = np.reshape(predicted,-1)

slope, intercept, R, p_value, std_err = stats.linregress(expected, predicted)
R2 = R*R
R2= round(R2, 3)
slope = round(slope, 3)
intercept = round(intercept, 3)

if intercept < 0:
    lin_r = 'y = '+str(slope)+'x'+str(intercept)
else:
    lin_r = 'y = '+str(slope)+'x+'+str(intercept)


print('Test data analysis')
print(lin_r)
print('R^2: ', R2)
print('mse: ', mse)


plt.plot(expected, predicted, 'bo', expected, expected, 'r', expected, regr.predict(expected[:, np.newaxis]), 'g')
plt.xlabel('expected')
plt.ylabel('predicted')
plt.savefig('test.png')
plt.close()

# Save the entire model as a SavedModel.
NN.save('NN_model/model')

# Delete NN model
tf.keras.backend.clear_session()
