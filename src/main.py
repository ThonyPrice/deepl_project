# Deep learning project DD2424
# W. Skagerstrom, N. Lindqvist, T. Price
# 2018-05-21

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
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from models import *

# Global parameters
EPOCHS = 20
BATCH_SIZE = 50
NUM_CLASSES = 200
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

def plotLoss(trainedModel, model_name):
    ''' Plots the loss function '''
    pyplot.switch_backend('agg')
    pyplot.plot(trainedModel.history['loss'])
    pyplot.plot(trainedModel.history['val_loss'])
    pyplot.title('model loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend(['train', 'validation'], loc='best')
    pyplot.legend(['train loss'], loc='upper left')
    pyplot.savefig('loss_plots/' + model_name)
    pyplot.close()


def getPreparedData():
    ''' Load and prepare train- and validation data '''
    X_tr, y_tr = getData.get_train_data()
    X_tr = X_tr.reshape(np.shape(X_tr)[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    y_tr = keras.utils.to_categorical(y_tr, NUM_CLASSES)
    X_val, y_val = getData.get_val_data()
    X_val = X_val.reshape(np.shape(X_val)[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
    return (X_tr, y_tr, X_val, y_val)


def trainModel(model_list):
    ''' Train all models in list and save their progress '''
    X_tr, y_tr, X_val, y_val = getPreparedData()
    # Prepare data generator object
    datagen = image_utils.ImageDataGenerator(
        featurewise_center=True, 
        samplewise_center=True
    )
    datagen.fit(X_tr)
    for m_name, params in model_list:
        print('Evaluating model: ' + m_name)
        model = getModel(*params)
        networkHistory = model.fit_generator(
            datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_tr) / BATCH_SIZE, epochs=EPOCHS,
            verbose=1, callbacks=None, validation_data=(X_val, y_val),
            use_multiprocessing=False, shuffle=False)
        plotLoss(networkHistory, m_name)
        scores = model.evaluate(X_val, y_val, verbose=1)
        print('Scores:')
        print(scores)
        with open('trainHistoryDict/' + m_name + '.pickle', 'wb') as file_pi:
            pickle.dump(networkHistory.history, file_pi)
        with open('scores/' + m_name + '.pickle', 'wb') as file_pi:
            pickle.dump(scores, file_pi)
        backend.clear_session()

def main():
    getData # Prepare data
    model_list = [
        # Name,             ["architect", (res),   "solver",  BN?, dropout, init
        # ("sgd",             ["vgg_z", (64, 64, 3), "sgd",     False,   0, 'random_uniform']),
        # ("sgd_mom",         ["vgg_z", (64, 64, 3), "sgd_mom", False,   0, 'random_uniform']),
        # ("adam_bn",         ["vgg_z", (64, 64, 3), "Adam",    True,    0, 'random_uniform']),
        # ("adam_drop",     ["vgg_z", (64, 64, 3), "Adam",    False, 0.3, 'random_uniform']),
        ("adam_bn_drop",    ["vgg_z", (64, 64, 3), "Adam",    True,  0.7, 'random_uniform']),
        ("adam_bn_drop_he", ["vgg_z", (64, 64, 3), "Adam",    True,  0.7, 'he_normal'])
    ]
    trainModel(model_list)


if __name__ == "__main__":
    main()
