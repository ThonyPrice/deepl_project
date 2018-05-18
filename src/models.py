from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,  Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing import image as image_utils

def getModel(modelString, dim):
    allModels = ['vgg_z']

    if (modelString == "vgg_z"):
        return vgg_z(dim)

    elif (modelString == "big_cnn_model"):
        return big_cnn_model(dim)



def vgg_z(dim):
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


def vgg_net16b(dim):
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
