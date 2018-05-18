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

from models import *


'''Plots the loss function'''
def plotLoss(trainedModel):
    pyplot.plot(trainedModel.history['loss'])
    pyplot.plot(trainedModel.history['val_loss'])
    pyplot.title('model loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend(['train', 'validation'], loc='uptrainLengthper left')
    pyplot.legend(['train loss'], loc='upper left')
    pyplot.show()


'''Fetches the dataset'''
def generateData(n_train, n_val):
    training_images, training_labels_encoded, \
    val_images, val_labels_encoded = getData.main(n_train, n_val)

    return training_images, training_labels_encoded, val_images, val_labels_encoded

def trainModel(epochs_n, model_list):
    # #X pictures, Y classes.
    # X_train, Y_train, X_val, Y_val = generateData(n_train, n_val)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_val.shape)
    # print(Y_val.shape)
    # X_train=X_train.reshape(n_train*200, 64,64,3)
    # X_val=X_val.reshape(n_val, 64,64,3)
    # Y_train = keras.utils.to_categorical(Y_train, 200)
    # Y_val = keras.utils.to_categorical(Y_val, 200)
    # X_train = np.divide(X_train, 255)
    # X_val = np.divide(X_val, 255)
    # print(X_train.shape)
    # print(X_val.shape)

    for m_name, params in model_list:
        model = getModel(*params)
        # Get and reshape validation data
        X_val, y_val = getData.get_val_data()
        X_val = X_val.reshape(getData.BATCH_SIZE, 64,64,3)
        y_val = keras.utils.to_categorical(y_val, 200)
        # Train iteratively
        for epoch in range(epochs_n):
            print("[PARENT EPOCH] epoch {}...".format(epoch + 1))
            for batch in getData.get_next_batch():
                X_batch, y_batch = batch[0], batch[1]
                print(np.shape(X_batch))
                X_batch = X_batch.reshape(np.shape(X_batch)[0], 64,64,3)
                y_batch = keras.utils.to_categorical(y_batch, 200)
                model.train_on_batch(X_batch, y_batch)
                model.test_on_batch(X_val, y_val)
        
        '''Use when using default mean of training'''
        networkHistory = network.fit(X_train, Y_train, verbose=1, epochs=epochs_n, batch_size=batchsize, callbacks=None, validation_data=[X_val, Y_val], shuffle=True)

        '''Use when using data augmentation'''
        #networkHistory = network.fit_generator(dataGenerator.flow(X_train, Y_train, batch_size=batchsize), steps_per_epoch=n_train/batchsize, epochs=epochs_n)



        with open('trainHistoryDict/' + m_name + '.pickle', 'wb') as file_pi:
            pickle.dump(networkHistory.history, file_pi)


        '''Plots the loss function of test and validation'''
        #plotLoss(networkHistory)


        '''Generates class predictions(if doing things manually, etc for ensemble)'''
        #predictions = network.predict(X_test, batch_size=100, verbose=0)


        '''Generates class predictions and checks accuracy'''
        scores = network.evaluate(X_val, Y_val, verbose=1)


def main():
    getData # Prepare data
    epochs = 2
    model_list = [
        # Name,             ["architect", (res),   "solver", BN?, dropout, init
        ("sgd",             ["vgg_z", (64, 64, 3), "sgd",  False,   0, 'random_uniform']),
        ("adam",            ["vgg_z", (64, 64, 3), "adam", False,   0, 'random_uniform']),
        ("adam_bn",         ["vgg_z", (64, 64, 3), "adam", True,    0, 'random_uniform']),
        ("adam_drop",       ["vgg_z", (64, 64, 3), "adam", False, 0.3, 'random_uniform']),
        ("adam_bn_drop",    ["vgg_z", (64, 64, 3), "adam", True,  0.3, 'random_uniform']),
        ("adam_bn_drop_he", ["vgg_z", (64, 64, 3), "adam", True,  0.3, 'he_normal'])
    ]
    trainModel(epochs, model_list)


if __name__ == "__main__":
    main()
