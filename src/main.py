#Deep learning project DD2424
#Names
#Date last modified

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,  Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import keras
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sqrt
import statistics
from os import listdir
from keras import callbacks, backend
from pandas import read_csv
import os
import sys
import matplotlib.pyplot as pyplot
import getData
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
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
def trainModel(n_train, n_val, epochs_n, batchsize):

    #X pictures, Y classes.
    X_train, Y_train, X_val, Y_val = generateData(n_train, n_val)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)


    X_train=X_train.reshape(n_train, 64,64,3)
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




    network = getModel("vgg_z", (64, 64, 3))


    '''Use when using default mean of training'''
    networkHistory = network.fit(X_train, Y_train, verbose=1, epochs=epochs_n, batch_size=batchsize, callbacks=None, validation_data=[X_val, Y_val], shuffle=True)



    '''Use when using data augmentation''''
    #networkHistory = network.fit_generator(dataGenerator.flow(X_train, Y_train, batch_size=batchsize), steps_per_epoch=n_train/batchsize, epochs=epochs_n)



    '''Plots the loss function of test and validation'''
    #plotLoss(networkHistory)



    '''Generates class predictions'''
    #predictions = network.predict(X_test, batch_size=100, verbose=0)



    '''Generates class predictions and checks accuracy'''
    scores = network.evaluate(X_val, Y_val, verbose=1)



def main():

    '''n_train is the number of images per class. Max 500. Total training samples = n_train * 200.'''
    n_train = 500

    '''n_val is the total number of validation images. Max 10000.'''
    n_val = 10000


    epochs = 10
    batchsize = 100
    trainModel(n_train, n_val, epochs, batchsize)





if __name__ == "__main__":
    main()
