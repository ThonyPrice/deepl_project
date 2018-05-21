from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,  Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing import image as image_utils
from keras.optimizers import Adam, SGD
from keras import metrics, regularizers

l2_rate = 0.001

def getModel(modelString, dim, opt = "adam", BN = False, dropout = 0, initializer = 'random_uniform'):
    allModels = ['vgg_z']

    if (modelString == "vgg_z"):
        return vgg_z(dim, opt, BN, dropout, initializer)

    elif (modelString == "big_cnn_model"):
        return big_cnn_model(dim)



def vgg_z(dim, opt, BN, dropout, initializer):
    model = Sequential()
    if opt == 'Adam':
        opt = Adam(lr=0.001)
    if opt == 'sgd_mom':
        opt = SGD(momentum=0.9, decay=0.99)
    if BN:
        do_BN = lambda : model.add(BatchNormalization())
    else:
        do_BN = lambda : None

    #Conv layers, round 1
    model.add(Conv2D(32, (2,2), padding="same", input_shape=dim,
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)
        ))
    do_BN()
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2,1), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)
        ))
    do_BN()
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)
        ))
    do_BN()
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Conv layers, round 2
    model.add(Conv2D(48, (2,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))
    do_BN()
    model.add(Activation('relu'))

    model.add(Conv2D(48, (2,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))
    do_BN()
    model.add(Activation('relu'))

    model.add(Conv2D(48, (2,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))
    do_BN()
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Conv layers, round 3
    model.add(Conv2D(80, (2,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))
    do_BN()
    model.add(Activation('relu'))

    model.add(Conv2D(80, (2,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))
    do_BN()
    model.add(Activation('relu'))

    model.add(Conv2D(80, (2,2), padding="same",
        kernel_initializer = initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))
    do_BN()
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048,
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(l2_rate)))

    do_BN()
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(200, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy', metrics.top_k_categorical_accuracy])
    return model







#The network model to be used
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
