from os import listdir
import numpy
import matplotlib.pyplot as pyplot
import pickle
import os
import matplotlib.ticker



def plotLoss(history, model_name, fontsize=12, titlefontsize = 14, xaxis=None):
    ''' Plots the loss function '''
    x_len = len(history['top_k_categorical_accuracy'])
    pyplot.switch_backend('agg')
    xaxis = [i for i in range(1,len(history['loss']))] + [len(history['loss'])]
    pyplot.plot(xaxis,history['loss'])
    pyplot.plot(xaxis,history['val_loss'])
    pyplot.title('Model Loss', fontsize=titlefontsize)
    pyplot.xlabel('epoch',fontsize=fontsize)
    pyplot.ylabel('loss',fontsize=fontsize)
    pyplot.legend(["Train loss", "Validation loss"], loc='best', fontsize=fontsize)

    locator = matplotlib.ticker.MultipleLocator(2)
    pyplot.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    pyplot.gca().xaxis.set_major_formatter(formatter)
    pyplot.xlim(0, x_len+1)
    pyplot.xticks(numpy.arange(0, x_len+1, 5.0))

    pyplot.savefig('testPlots/lossPlots/' + model_name+"_loss")
    pyplot.close()

def plotTop1Acc(history, model_name,fontsize=12,titlefontsize = 14, xaxis=None):
    ''' Plots the acc function '''
    x_len = len(history['top_k_categorical_accuracy'])
    pyplot.switch_backend('agg')
    xaxis = [i for i in range(1,len(history['acc']))] + [len(history['loss'])]
    pyplot.plot(xaxis,history['acc'])
    pyplot.plot(xaxis,history['val_acc'])
    pyplot.title('Top 1 Model Accuracy', fontsize=titlefontsize)
    pyplot.xlabel('Epoch', fontsize=fontsize)
    pyplot.ylabel('Accuracy', fontsize=fontsize)
    pyplot.legend(["Train accuracy", "Validation accuracy"], loc='best', fontsize=fontsize)

    locator = matplotlib.ticker.MultipleLocator(2)
    pyplot.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    pyplot.gca().xaxis.set_major_formatter(formatter)
    pyplot.xlim(0, x_len+1)
    pyplot.xticks(numpy.arange(0, x_len+1, 5.0))


    pyplot.savefig('testPlots/accPlots/' + model_name+"_top1")
    pyplot.close()

def plotTop5Acc(history, model_name,fontsize=12,titlefontsize = 14, xaxis=None):
    ''' Plots the acc function '''
    x_len = len(history['top_k_categorical_accuracy'])
    pyplot.switch_backend('agg')
    xaxis = [i for i in range(1,len(history['top_k_categorical_accuracy']))] + [len(history['top_k_categorical_accuracy'])]
    pyplot.plot(xaxis,history['top_k_categorical_accuracy'])
    pyplot.plot(xaxis,history['val_top_k_categorical_accuracy'])
    pyplot.title('Top 5 Model Accuracy', fontsize=titlefontsize)
    pyplot.xlabel('Epoch', fontsize=fontsize)
    pyplot.ylabel('Accuracy', fontsize=fontsize)
    pyplot.legend(["Train accuracy", "Validation accuracy"], loc='best', fontsize=fontsize)

    locator = matplotlib.ticker.MultipleLocator(2)
    pyplot.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    pyplot.gca().xaxis.set_major_formatter(formatter)
    pyplot.xlim(0, x_len+1)
    pyplot.xticks(numpy.arange(0, x_len+1, 5.0))


    pyplot.savefig('testPlots/top5accPlots/' + model_name+"_top5")
    pyplot.close()


def main():
    for type in listdir('./trainHistoryDict/'):
        with open('./trainHistoryDict/'+str(type), 'rb') as file:

            history = pickle.load(file)
            name = ((os.path.splitext(type))[0])

            plotLoss(history, name)
            plotTop1Acc(history, name)
            plotTop5Acc(history, name)

        file.close()







if __name__ == "__main__":
    main()
