import time
import csv;
import numpy as np;
from scipy.special import expit as sig
import matplotlib as plt

def SSE(delta):
    return np.sum(delta ** 2)


def sigprime(activation):

        return activation - activation**2 # a(1-a)

def relu(x):
    return [i if i>0 else 0 for i in x]


def reluprime(x):
    return [1 if i>0 else 0 for i in x]
flag = False
mapping = {
"0" :48,
"1" :49,
"2" :50,
"3" :51,
"4" :52,
"5" :53,
"6" :54,
"7" :55,
"8" :56,
"9" :57,
"10": 65,
"11": 66,
"12": 67,
"13": 68,
"14": 69,
"15": 70,
"16": 71,
"17": 72,
"18" :73,
"19" :74,
"20": 75,
"21": 76,
"22" :77,
"23" :78,
"24" :79,
"25" :80,
"26" :81,
"27" :82,
"28" :83,
"29" :84,
"30" :85,
"31" :86,
"32" :87,
"33" :88,
"34" :89,
"35" :90,
"36" :97,
"37" :98,
"38" :100,
"39" :101,
"40" :102,
"41" :103,
"42" :104,
"43" :110,
"44" :113,
"45" :114,
"46" :116,
}
class ANN:

    def __init__(self, architecture):

        self.layers = len(architecture)-1
        self.architecture = architecture
        self.updateMatrices = []
        self.weightMatrices = self.initializeWeightMatrices()
        self.guessFrequency = []
        self.inputs = []
        self.outputs = []
        self.deltas = []
        self.trainingSet = []
        self.trainingLabels = []

    def initializeWeightMatrices(self):
        matrices = []
        for (prev, next) in zip(self.architecture[:-1], self.architecture[1:]):
            self.updateMatrices.append(np.zeros(shape = (next,prev +1)))
            matrices.append(np.random.normal(scale=0.01 ,size = (next,prev +1))) #plus one for bias
        return matrices

    def forwardPass(self, digit):

        self.activations = []  # activations for layers i.e layer inputs
        self.outputs = []       # sig(activations) for layers i.e layer output

        digit = np.append(digit,1) # Augment input

        digit = np.transpose(digit)
        self.activations.append(digit) # a1 = augmented digit
        for i in range(self.layers):

            if i == 0:
                z = self.weightMatrices[0].dot(self.activations[-1]) # z2 = w1 * a1

            else:



                z = self.weightMatrices[i].dot(self.activations[-1]) # zx = w(x-1)*a(x-1)
            unaugmented_activation = relu(z)
            augmented_activation = np.append(unaugmented_activation,1)
            if i != self.layers -1:
                self.activations.append(augmented_activation)
            else:
                normalizer = sum(unaugmented_activation) +1
                unaugmented_activation = [i/normalizer for i in unaugmented_activation]

                self.activations.append(unaugmented_activation)

        #print("activation shapes")
        #for i in self.activations:
        #    print(i.shape)
        return self.activations[-1]
    def backpropagation(self,digit,digit_label):

        self.deltas =[]



        self.forwardPass(digit)

        # Compute Deltas i.e distribute blame for error across nodes in the network

        for i in reversed(range(self.layers)):

            if i == self.layers - 1:             # if calculating output delta of the network


                delta = self.activations[-1] - np.transpose(digit_label)
                #print("error is:")
                error = SSE(delta)
                #print(error)
                self.deltas.append(delta)
            else:
                delta = np.transpose(self.weightMatrices[i+1]).dot(self.deltas[-1])

                self.deltas.append(delta[:-1:] * reluprime(self.activations[i+1][:-1:]))  # dont care about delta on bias so truncate delta

        # Deltas are in reverse order so reverse again

        self.deltas = self.deltas[::-1]
        for i in range(len(self.deltas)):
            self.deltas[i] = np.reshape(self.deltas[i], (len(self.deltas[i]),1))

        #print("delta shapes")
        #for delta in self.deltas:
        #    print(delta.shape)

        for i in range(len(self.activations)):

            self.activations[i] = np.reshape(self.activations[i],(len(self.activations[i]),1))
        self.computeGradients()
        self.updateWeights("sgd")
        return error
    def computeGradients(self):



        for i in range(len(self.updateMatrices)):

        #    print('D^{0} = d^{1}*a^{2}'.format(i,i+1,i))
        #    print('{0} = {1}*{2}'.format(updateMatrices[i].shape, self.deltas[i].shape, np.transpose(self.activations[i]).shape))

            self.updateMatrices[i] = np.matmul(self.deltas[i],np.transpose(self.activations[i]))

        #    print(updateMatrices[i].shape)
        #for i in updateMatrices:
        #    print(i.shape)
        #print("==================")



    def updateWeights(self,optimizer):

        for i in range(len(self.weightMatrices)):
            alpha = 0.001
            self.weightMatrices[i] += -alpha * self.updateMatrices[i]
      #      print(i)
       #     print(self.weightMatrices[i].shape)
        #    print(updateMatrices[i].shape)






    def readTrainingSet(self,filenumber):
        self.trainingSet = []
        self.trainingLabels = []
        filename = "emnist-balanced-train" + str(filenumber) + '.csv'
        trainingset = np.loadtxt(open(filename, "rb"), delimiter=",")


        self.trainingLabels = []
        for i in range(len(trainingset)):
            label = np.zeros(47)

            label[int(trainingset[i][0])] = 1
            self.trainingLabels.append(label)

            self.trainingSet.append(trainingset[i][1:])

    def readTestSet(self,filenumber):


        testLabels = []
        filename = "emnist-balanced-test" + str(filenumber) + ".csv"
        testSet = np.loadtxt(open(filename, "rb"), delimiter=",")

        for i in range(len(testSet)):
            label = np.zeros(47)
            label[int(testSet[i][0])-1] = 1
            testLabels.append(label)

        testSet = [x[1:] for x in testSet]

        testDict = {"testSet": testSet,
                    "testLabels": testLabels}

        return testDict

    def checkWeightMatrixDimensions(self):
        print("shapes")
        for matrix in self.weightMatrices:

            print(matrix.shape)

    def trainBatch(self):
        total_error = 0
        for i in range(len(self.trainingSet)):
            total_error += self.backpropagation(self.trainingSet[i],self.trainingLabels[i])
        return total_error
    def getLabelDigit(self, digit_label):
        i = np.where(digit_label == 1)

        return chr(mapping[str(i[0][0])])
    def getGuessDigit(self, guess):
        return chr(mapping[str(guess)])
    def checkDigit(self,digit,digit_label):



        #print("digit is:")

        label=self.getLabelDigit(digit_label)
        #print(label)

        self.forwardPass(digit)


        #print("guess is:")
        guessLabel = np.argmax(self.activations[-1])
        self.guessFrequency[guessLabel] += 1
        guess= self.getGuessDigit(guessLabel)
        #print(guess)
        if(guess == label):

            return 1
        return 0
    def test(self):
        self.guessFrequency = np.zeros(47)
        total_digits = 0
        correct = 0
        for i in range(19):
            test =self.readTestSet(i)
            total_digits += len(test["testSet"])
            for i in range(len(test["testSet"])):
                correct+= self.checkDigit(test["testSet"][i], test["testLabels"][i])
        accuracy = (correct/total_digits) *100
        print("accuracy is: {0}%".format(accuracy))
        return accuracy

    def small_test(self,i):
        self.guessFrequency = np.zeros(47)
        total = 0
        error = 0
        count = 0
        for i in range(i):
            testset = self.readTestSet(i)
            total += len(testset["testSet"])
            for i in range(len(testset["testSet"])):
                label =testset["testLabels"][i]
                network_output = self.forwardPass(testset["testSet"][i])
                error += SSE(network_output - label)
                guessLabel = np.argmax(network_output)
                self.guessFrequency[guessLabel] += 1
                guess = self.getGuessDigit(guessLabel)


                labelDigit = self.getLabelDigit(label)
                if(guess == labelDigit):
                    count +=1
        accuracy = (count/total) *100
        print(self.guessFrequency)
        return error, accuracy

ann =ANN([784,300,300,47])

ann.checkWeightMatrixDimensions()

for j in range(10):

    for i in range(10):

        ann.readTrainingSet(i)
        #print("training batch: {0} \n".format(i))


        ann.trainBatch()



         #   print("at batch {0}\n".format(i))
          #  ann.test()

        #   flag = False
    print("small test error after epoch {0} is {1}".format(j+1,ann.small_test(1)))
print(ann.test())
print(ann.guessFrequency)