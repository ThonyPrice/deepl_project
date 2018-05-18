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


# The network model to be used
def network_model(dim):
    model = Sequential()
    model.add(Dense(40, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def big_cnn_model(dim):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=dim, activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(128, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(256, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(256, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(512, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(512, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(512, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(512, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096 , activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096 , activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    return model


def cnn_model(dim):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=dim, activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(Conv2D(128, (3, 3), padding="same",activation='relu'))
    model.add(Conv2D(128, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000 , activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1000 , activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(200, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    return model


def vgg_net(dim):
    model = Sequential()

    #Conv layers, round 1
    model.add(Conv2D(32, (2,2), padding="same", input_shape=dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (2,1), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (1,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Conv layers, round 2
    model.add(Conv2D(48, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(48, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(48, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Conv layers, round 3
    model.add(Conv2D(80, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(80, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(80, (2,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def vgg_net16(dim):
    model = Sequential()

    #Conv layers, round 1
    model.add(Conv2D(64, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(64, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(128, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(256, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(256, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(512, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(512, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(512, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(Conv2D(512, (3,3), padding="same", input_shape=dim, activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(200, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def new_mode(dim):
    return model


# Trains the specified model
def trainModel(n_pictures, epochs_n, batchsize):

    #X pictures, Y classes.
    X_train, Y_train, X_val, Y_val = generateData(n_pictures)


    X_train=X_train.reshape(4000, 64,64,3)
    X_val=X_val.reshape(20, 64,64,3)

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


    network = vgg_net16((64, 64, 3))
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

    n_pictures = 10
    epochs = 1
    batchsize = 5
    trainModel(n_pictures, epochs, batchsize)





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
