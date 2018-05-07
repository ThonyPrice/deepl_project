#Deep learning project DD2424
#Names
#Date last modified

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
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


#Plots the loss function
def plotLoss(trainedModel):
    pyplot.plot(trainedModel.history['loss'])
    pyplot.plot(trainedModel.history['val_loss'])

    pyplot.title('model loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')

    pyplot.legend(['train', 'validation'], loc='uptrainLengthper left')
    pyplot.legend(['train loss'], loc='upper left')

    pyplot.show()


#Fetches the dataset
def generateData(n_pictures):
    #TODO Get picture data
    training_images, training_labels_encoded, \
        val_images, val_labels_encoded = getData.main()


    return training_images, training_labels_encoded, val_images, val_labels_encoded
    #print(training_images)
    #sys.exit(0)


#The network model to be used
def network_model(dim):
    model = Sequential()
    model.add(Dense(40, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Trains the specified model
def trainModel(n_pictures, epochs_n=30, batchsize=100):

    #X pictures, Y classes.
    X_train, Y_train, X_val, Y_val = generateData(n_pictures)

    Y_train = keras.utils.to_categorical(Y_train, 200)
    Y_val = keras.utils.to_categorical(Y_val, 200)
    print(Y_train)

    #print(X_train.shape)
    #print(Y_train.shape)

    #X_val = X_val.T
    #X_train = X_train.T


    #TODO Scale the dataset. Some scaling options: See imports from sklearn.preprocessing
    #TODO Might need to reshape the data

    earlyStop = callbacks.EarlyStopping(monitor='accuracy', min_delta=0.00005, patience=2000, verbose=0, mode='max')

    #print(X_train[:,0].shape)
    network = network_model(X_train[0,:].shape[0])
    X_train = np.divide(X_train, 255)
    X_val = np.divide(X_val, 255)

    networkHistory = network.fit(X_train, Y_train, verbose=1, epochs=epochs_n, batch_size=batchsize, callbacks=[earlyStop], validation_data=[X_val, Y_val], shuffle=True)

    #Plots the loss function of test and validation
    #plotLoss(networkHistory)

    #Generates class predictions
    #predictions = network.predict(X_test, batch_size=100, verbose=0)


    #Generates class predictions and checks accuracy
    scores = network.evaluate(X_val, Y_val, verbose=1)
    #print("%s: %.2f%%" % (network.metrics_names[1], network[1]*100))




def main():

    n_pictures = 500
    epochs = 200
    batchsize = 100
    trainModel(n_pictures, epochs, batchsize)





if __name__ == "__main__":
    main()
