'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
import keras.backend as K
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import adagrad
from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import time

epochs = 1
trainingResults = {}
trainingSize = 60000
number_classes = 10
graphSamplePeriod = 100
tableSamplePeriod = 6000
trainAveragingIterations = 10
batch_size = 100

resultsTable = {}
"""Calls Backs"""
class IterationSampler(Callback):

        def on_train_begin(self, logs=None):
            self.currentCount = 0
            self.graphSamplePeriod = graphSamplePeriod
            self.testAccuracy = []
            self.testLoss = []
            trainingResults[self.model.identifier] = {"training_accuracy": [], "training_loss":[], "test_accuracy": [], "test_loss": []}
        def on_batch_end(self, batch, logs={}):
            self.currentCount += logs.get('size')
            if self.currentCount % self.graphSamplePeriod == 0:

                trainingResults[self.model.identifier]['training_accuracy'].append(logs.get('acc'))
                trainingResults[self.model.identifier]['training_loss'].append(logs.get('loss'))
            if self.currentCount %tableSamplePeriod ==0:
                loss,accuracy = self.model.evaluate(test_digits,test_labels,verbose=0)
                self.testAccuracy.append(accuracy)
                self.testLoss.append(loss)
        def on_train_end(self, logs=None):
            trainingResults[self.model.identifier]['test_accuracy'] = self.testAccuracy
            trainingResults[self.model.identifier]['test_loss'] = self.testLoss

class SearchThenConverge(Callback):
    def __init__(self,T0,LR):
        self.T0 = T0
        self.a0 = LR
        self.currentCount = 0.0


    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.model.optimizer.lr, self.getLR())


    def on_batch_end(self, batch, logs=None):
        self.currentCount += logs.get('size')
    def getLR(self):

        lr = self.a0 / (1 + (self.currentCount /self.T0)) # converge after searching for 90% of epoch

        return lr
class CyclicalLearningRates(Callback):

    def __init__(self,a_min,a_max):
        self.stepsize = 2 * trainingSize
        self.a_min = a_min
        self.a_max = a_max

    def on_train_begin(self, logs=None):
        self.currentCount = 0
    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.model.optimizer.lr, self.getLR())
    def getLR(self):
        cycleLength = np.floor(1 + (self.currentCount/2*self.stepsize))     # floor(1 + iteration/(2*step size))
        x = np.abs((self.currentCount/self.stepsize) - 2*cycleLength + 1)
        return self.a_min + (self.a_max-self.a_min)*((1-x)*(x < 1))     # a = amin +(amax - amin)*max{0,1-x}


"""Experimental Data Analysis"""
def CLR_test():
    accs = []
    lrs = [0.01*(i+1) for i in range(0,12)]
    for i in range(12):
        print(i)
        model = Sequential()

        model.add(Dense(300, activation='relu', input_shape=(784,)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(number_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=lrs[i]),
                      metrics=['accuracy'])

        history = model.fit(training_digits, training_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(test_digits, test_labels),
                            )
        score = model.evaluate(validation_digits, validation_labels, verbose=0)
        accs.append(score[1])
    plt.plot(lrs,accs)
    plt.show()

def STC_test():

    T0 = [100*(i+1) for i in range(2230,2250)]
    accs = []

    for i in T0:
        print(i)
        stc= SearchThenConverge(i,0.01)
        model = Sequential()

        model.add(Dense(300, activation='relu', input_shape=(784,)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(number_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(),
                      metrics=['accuracy'])

        history = model.fit(training_digits, training_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(test_digits, test_labels),
                            callbacks=[stc]
                            )
        score = model.evaluate(validation_digits, validation_labels, verbose=0)
        accs.append(score[1])
    plt.plot(T0,accs)
    plt.show()


"""Read and prepare Training Set"""
(training_digits, training_labels), (test_digits, test_labels) = mnist.load_data()

training_digits = training_digits.reshape(60000, 784)
test_digits = test_digits.reshape(10000, 784)
training_digits = training_digits.astype('float32')
test_digits = test_digits.astype('float32')
training_digits /= 255
test_digits /= 255
print(training_digits.shape[0], 'train samples')
print(test_digits.shape[0], 'test samples')

# convert class vectors to binary class matrices
training_labels = keras.utils.to_categorical(training_labels, number_classes)
test_labels = keras.utils.to_categorical(test_labels, number_classes)
validation_digits = test_digits[0:100]
test_digits = test_digits[100:]
validation_labels = test_labels[0:100]
test_labels = test_labels[100:]

"""Training Nets"""
def train(optimizer, identifier,verbose, callbacks,iterations):

    """Graphing results"""
    accuracy = np.zeros(int(trainingSize/graphSamplePeriod))
    loss = np.zeros(int(trainingSize/graphSamplePeriod))

    """Table Results"""

    test_accuracy = np.zeros(int(trainingSize/tableSamplePeriod))
    test_loss = np.zeros(int(trainingSize/tableSamplePeriod))

    final_acc = 0
    for i in range(iterations):
        print("{0},{1}".format(identifier,i))
        model = Sequential()
        model.identifier = identifier
        model.add(Dense(300, activation='relu', input_shape=(784,)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(number_classes, activation='softmax'))


        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        history = model.fit(training_digits, training_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(test_digits, test_labels),
                            callbacks=callbacks)

        accuracy += trainingResults[identifier]["training_accuracy"]
        loss += trainingResults[identifier]["training_loss"]
        test_accuracy += trainingResults[identifier]["test_accuracy"]
        test_loss += trainingResults[identifier]["test_loss"]

    accuracy /= iterations
    loss /= iterations
    test_loss /= iterations
    test_accuracy /= iterations
    return {"training_accuracy": accuracy, "training_loss": loss, "test_accuracy": test_accuracy, "test_loss":test_loss}

"""Declare Callbacks"""
sample = IterationSampler()
stc = SearchThenConverge(225000, 0.01)
clr = CyclicalLearningRates(0.02,0.12)

def Main():

    """Main Loop"""
    start = time.time()



    averagedResult = {}
    averagedResult["SGD"]=train(SGD(),"SGD",0, [sample],trainAveragingIterations)
    averagedResult["Momentum"]=train(SGD(momentum=0.9),"Momentum", 0,[sample],trainAveragingIterations)
    averagedResult["Search-then-Converge"]=train(SGD(), "STC",0, [sample,stc],trainAveragingIterations)
    averagedResult["Cyclical Learning Rate"]=train(SGD(), "CLR",0,[sample,clr],trainAveragingIterations)
    averagedResult["ADAGRAD"]=train(adagrad(),"Adagrad",0,[sample],trainAveragingIterations)
    averagedResult["RMSProp"]=train(RMSprop(),"RMSProp",0,[sample],trainAveragingIterations)
    averagedResult["Adam"]=train(adam(), "Adam",0,[sample],trainAveragingIterations)
    print("Time")
    print(time.time()-start)


    iterations = [graphSamplePeriod*(i+1) for i in range(int(trainingSize/graphSamplePeriod))]

    for i in averagedResult:
        accuracy = averagedResult[i]['training_accuracy']

        plt.plot(iterations, accuracy*100, label=i)
        plt.xlabel("Iteration")
        plt.ylabel("Training Accuracy (%)")
        plt.legend()
    plt.show()
    for i in averagedResult:
        loss = averagedResult[i]['training_loss']

        plt.plot(iterations, loss, label=i)
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.legend()
    plt.show()

    iterations = [tableSamplePeriod*(i+1) for i in range(int(trainingSize/tableSamplePeriod))]
    for i in averagedResult:
        test_accuracy = averagedResult[i]['test_accuracy']

        plt.plot(iterations, test_accuracy * 100, label=i)
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy (%)")
        plt.legend()
    plt.show()
    for i in averagedResult:
        loss = averagedResult[i]['test_loss']

        plt.plot(iterations, loss, label=i)
        plt.xlabel("Iteration")
        plt.ylabel("Test Loss")
        plt.legend()
    plt.show()
    for i in averagedResult:
        print("{0} test accuracy: {1} test loss: {2}".format(i,averagedResult[i]["test_accuracy"],averagedResult[i]["test_loss"]))

Main()