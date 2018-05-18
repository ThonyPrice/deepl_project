# Deep learning project DD2424
# W. Skagerstrom, N. Lindqvist, T. Price
#Date last modified

import getData
import keras
import os
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import statistics
import sys

from math import sqrt
from matplotlib import pyplot as plt
from os import listdir
from pandas import read_csv
from keras import callbacks, backend

from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten,
    BatchNormalization, AveragePooling2D, ZeroPadding2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D
)
from keras.preprocessing import image as image_utils
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold


# Plots the loss function
def plotLoss(trainedModel):
    pyplot.plot(trainedModel.history['loss'])
    pyplot.plot(trainedModel.history['val_loss'])
    pyplot.title('model loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend(['train', 'validation'], loc='uptrainLengthper left')
    pyplot.legend(['train loss'], loc='upper left')
    pyplot.show()


# Fetches the dataset
def generateData(n_pictures):
    training_images, training_labels_encoded, \
        val_images, val_labels_encoded = getData.main()
    return training_images, training_labels_encoded, val_images, val_labels_encoded


# Trains the specified model
def trainModel(n_train, n_val, epochs_n, batchsize):
    #X pictures, Y classes.
    X_train, Y_train, X_val, Y_val = generateData(n_train, n_val)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)

    X_train=X_train.reshape(n_train, 64,64,3)
    X_val=X_val.reshape(n_val, 64,64,3)

    #plt.imshow(X_train[0])
    #plt.show()

    Y_train = keras.utils.to_categorical(Y_train, 200)
    Y_val = keras.utils.to_categorical(Y_val, 200)

    #X_val = X_val.T
    #X_train = X_train.T

    earlyStop = callbacks.EarlyStopping(monitor='acc', min_delta=0.000005, patience=2000, verbose=0, mode='max')

    X_train = np.divide(X_train, 255)
    X_val = np.divide(X_val, 255)
    print(X_train.shape)
    print(X_val.shape)

    network = getModel("vgg_z", (64, 64, 3))
    networkHistory = network.fit(X_train, Y_train, verbose=1, epochs=epochs_n, batch_size=batchsize, callbacks=None, validation_data=[X_val, Y_val], shuffle=True)

    with open('/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(networkHistory.history, file_pi)

    #Plots the loss function of test and validation
    #plotLoss(networkHistory)

    #Generates class predictions
    #predictions = network.predict(X_test, batch_size=100, verbose=0)

    #Generates class predictions and checks accuracy
    scores = network.evaluate(X_val, Y_val, verbose=1)


def main():
    #n_train is the number of images per class. Max 500. Total training samples = n_train * 200.
    n_train = 500
    #n_val is the total number of validation images. Max 10000.
    n_val = 10000

    epochs = 10
    batchsize = 100
    trainModel(n_train, n_val, epochs, batchsize)


if __name__ == "__main__":
    main()


'''

from keras.layers import Conv2D, Input

# input tensor for a 3-channel 64x64 image
x = Input(shape=(64, 64, 3))
# 3x3 conv with 3 output channels (same as input channels)
y = Conv2D(3, (3, 3), padding='same')(x)
# this returns x + y.
z = keras.layers.add([x, y])


'''
