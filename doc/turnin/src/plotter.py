from os import listdir
import numpy
import matplotlib.pyplot as pyplot
import pickle
import os
import matplotlib.ticker



def plotLoss(history, model_name, fontsize=12, titlefontsize = 14, xaxis=None):
    ''' Plots the loss function '''

    pyplot.switch_backend('agg')
    xaxis = [i for i in range(1,len(history['loss']))] + [len(history['loss'])]
    pyplot.plot(history['loss'])
    pyplot.plot(history['val_loss'])
    pyplot.title('Model Loss', fontsize=titlefontsize)
    pyplot.xlabel('epoch',fontsize=fontsize)
    pyplot.ylabel('loss',fontsize=fontsize)
    pyplot.legend(["Train loss", "Validation loss"], loc='best', fontsize=fontsize)


    pyplot.savefig('testPlots/lossPlots/' + model_name+"_loss")
    pyplot.close()

def plotTop1Acc(history, model_name,fontsize=12,titlefontsize = 14, xaxis=None):
    ''' Plots the acc function '''

    pyplot.switch_backend('agg')
    xaxis = [i for i in range(1,len(history['acc']))] + [len(history['loss'])]
    pyplot.plot(history['acc'])
    pyplot.plot(history['val_acc'])
    pyplot.title('Top 1 Model Accuracy', fontsize=titlefontsize)
    pyplot.xlabel('Epoch', fontsize=fontsize)
    pyplot.ylabel('Accuracy', fontsize=fontsize)
    pyplot.legend(["Train accuracy", "Validation accuracy"], loc='best', fontsize=fontsize)





    pyplot.savefig('testPlots/accPlots/' + model_name+"_top1")
    pyplot.close()

def plotTop5Acc(history, model_name,fontsize=12,titlefontsize = 14, xaxis=None):
    ''' Plots the acc function '''

    pyplot.switch_backend('agg')
    xaxis = [i for i in range(1,len(history['top_k_categorical_accuracy']))] + [len(history['top_k_categorical_accuracy'])]
    pyplot.plot(history['top_k_categorical_accuracy'])
    pyplot.plot(history['val_top_k_categorical_accuracy'])
    pyplot.title('Top 5 Model Accuracy', fontsize=titlefontsize)
    pyplot.xlabel('Epoch', fontsize=fontsize)
    pyplot.ylabel('Accuracy', fontsize=fontsize)
    pyplot.legend(["Train accuracy", "Validation accuracy"], loc='best', fontsize=fontsize)



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

            print("model name: " + name)
            print(name + " top 1 acc: " + str(history['val_acc'][-1]))
            print(name + " top 5 acc: " + str(history['val_top_k_categorical_accuracy'][-1]))

        file.close()







if __name__ == "__main__":
    main()
