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



'''Trains the specified model'''
def trainModel(n_train, n_val, epochs_n, batchsize, model_list):
    #X pictures, Y classes.
    X_train, Y_train, X_val, Y_val = generateData(n_train, n_val)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)

    X_train=X_train.reshape(n_train*200, 64,64,3)
    X_val=X_val.reshape(n_val, 64,64,3)


    Y_train = keras.utils.to_categorical(Y_train, 200)
    Y_val = keras.utils.to_categorical(Y_val, 200)




    X_train = np.divide(X_train, 255)
    X_val = np.divide(X_val, 255)
    print(X_train.shape)
    print(X_val.shape)




    '''Data generator for data agumentation'''
    #dataGenerator = image.ImageDataGenerator()
    #dataGenerator.fit(X_train)

    for m_name, params in model_list:

        network = getModel(*params)


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

    '''n_train is the number of images per class. Max 500. Total training samples = n_train * 200.'''
    n_train = 3

    '''n_val is the total number of validation images. Max 10000.'''
    n_val = 10

    epochs = 2
    batchsize = 100
    model_list = [
        ("sgd", ["vgg_z", (64, 64, 3), "sgd", 0.3]),
        ("adam", ["vgg_z", (64, 64, 3), "adam", 0])
    ]
    trainModel(n_train, n_val, epochs, batchsize, model_list)


if __name__ == "__main__":
    main()
