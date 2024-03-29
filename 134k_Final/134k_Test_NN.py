# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:24:19 2021

@author: josep
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys

descriptor = str(sys.argv[1])
print("Loading testing data...")
    
x_test = np.load(str(descriptor) + "_x_test.npy", allow_pickle=True)
y_test = np.load(str(descriptor) + "_y_test.npy", allow_pickle=True)

checkpoint_filepath = "NN_model_best"
NN = tf.keras.models.load_model(checkpoint_filepath)

################### NN part ################################
import os
print("NN workflow...")
if GPU == "GPU":
    print("Using GPU...")

elif GPU == "CPU":

    # GPU OFF
    print("Using CPU with " + str(cores) + " core(s)...")
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
else:
    print("Invalid processor called. Type GPU or CPU as third argument.")
    exit()

def setup_NN(model, size_input, NN_top):

    # add input layer (input_shape=(size_input,)) and first hidden layer to NN
    model.add(tf.keras.layers.Dense(
        NN_top[0],
        input_shape=(size_input,),
     #   kernel_initializer = tf.keras.initializers.RandomUniform(minval= -2., maxval= 2., seed=None),
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=None),
        activation='softplus',
        use_bias=True))

    # add hidden layers to NN 
    for i in range(len(NN_top)-1):
        model.add(tf.keras.layers.Dense(
            NN_top[i+1],
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed=None),
            activation='softplus',
            use_bias=True))

    # add output layer 
    model.add(tf.keras.layers.Dense(
        1,
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=None),
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

NN_top.append(200)
NN_top.append(100)
NN_top.append(10)

setup_NN(NN, size_input, NN_top)

# NN build
n_epoch = 300

#CALLBACKS
#Finding best NN model during training process
checkpoint_filepath = "NN_model_best"
mc = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
    save_best_only=True)

#es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode = "min", verbose=1, patience = 50, restore_best_weights=True)
cb_list = [mc]

NN.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mean_squared_error'])
start = time.time()
history = NN.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = n_epoch, batch_size = 64, verbose = 1,callbacks=cb_list)
end = time.time()
print("Training the neural network for " + str(descriptor) + " with a " + str(GPU) + " and " + str(cores) + " core(s) took: {}".format(end-start))

# PLOT AND STATS RESULTS
import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['mean_squared_error'], label="train")
plt.plot(history.history['val_mean_squared_error'],label="val")
plt.title('model mse for ' +str(descriptor))
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', "val"], loc='upper left')
plt.savefig(str(descriptor) + '_train_and_val_process.png')
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
plt.savefig(str(descriptor) + '_train.png')
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
plt.savefig(str(descriptor) + '_test.png')
plt.close()

# Save the entire model as a SavedModel.
NN.save(str(descriptor)+'_NN_model/model')

# Delete NN model
tf.keras.backend.clear_session()
