import csv
import numpy
import re
import math

class GDA:
    def __init__(self, data):
        self.data = data
        #moving the data into 2 separate arrays based on their class
        #any items with a different class than 1 or -1 will trigger an error message
        group1 = []
        group2 = []
        for sample in data:
            if sample[-1] == 1:
                group1.append(sample)
            elif sample[-1] == -1:
                group2.append(sample)
            else:
                print('INVALID CLASS')
        #finding the number of samples in each class
        N1 = len(group1)
        N2 = len(group2)
        #computing the ratio samples in class 1
        pi = N1 / (N1 + N2)
        #computing the mean across all samples in class 1 by adding them by feature and then dividing each value by N1
        miu1 = group1[0][:-1]  # [:-1] so that class is not included
        for sample in group1[1:]:
            miu1 = [sample[i] + miu1[i] for i in range(len(miu1))]
        miu1 = [i / N1 for i in miu1]
        #computing the mean across all samples in class -1 by adding them by feature and then dividing each value by N2
        miu2 = group2[0][:-1]  # [:-1] so that class is not included
        for sample in group2[1:]:
            miu2 = [sample[i] + miu2[i] for i in range(len(miu2))]
        miu2 = [i / N2 for i in miu2]
        #empty arrays for covariance values to be appended to
        cov1 = [[] for i in range(len(miu1))]
        cov2 = [[] for i in range(len(miu2))]
        #filling covariance matrices with a 20*20 field of 0's
        for i in range(20):
            for j in range(20):
                cov1[i].append(0)
                cov2[i].append(0)
        #computing the vector difference between each value in either class and the class's mean
        #then does the vector outer product of the difference and adds this to the covariance matrix
        for sample in group1:
            temp = [sample[i] - miu1[i] for i in range(len(miu1))]
            cov1 += numpy.outer(temp,temp)
        for sample in group2:
            temp = [sample[i] - miu2[i] for i in range(len(miu2))]
            cov2 += numpy.outer(temp,temp)
        #computing the covariance matrix
        sigma = (cov1 + cov2) / (N1 + N2)
        #computing the inverse as it is needed in calculations
        sigInverse = numpy.linalg.inv(sigma)
        #computing the transpose of both mean vectors as they're needed in calculations
        miu1T = [list(x) for x in zip(miu1)]
        miu2T = [list(x) for x in zip(miu2)]
        #computing the log odds for the bias term
        logOdds = math.log(N1 / N2)
        #print('Mean 1: ' + str(miu1))
        #print('Mean 2: ' + str(miu2))
        #print('Covariance matrix: ' + str(sigma))
        #print('Log odds: ' + str(logOdds))
        #computing the slope coefficient of the LDA using the maximum likelihhod method
        self.slope = numpy.matmul(sigInverse, [list(x) for x in zip([miu1[i] - miu2[i] for i in range(len(miu1))])])
        #computing the bias coefficient of the LDA using the maximum likelihhod method
        self.bias = logOdds - (0.5 * numpy.matmul(numpy.matmul(miu1,sigInverse),miu1T)) + (0.5 * numpy.matmul(numpy.matmul(miu2,sigInverse),miu2T))

    def predict(self, x):
        results = []
        #looping through all the given values to make a prediction on each
        for givenData in x:
            #computing the value of alpha setby the given sample and the 2 coefficients of the model
            value = numpy.inner(numpy.transpose(self.slope), givenData[:-1]) + self.bias
            #computing the probaility of class 1 using the sigmoid funtion
            probability = 1 / (1 + math.exp(-1 * value))
            #setting the proper class based on the probability
            #tiebreaker = class -1
            if(probability > 0.5):
                results.append(1)
            else:
                results.append(-1)
        return results


if True:   #switch for reading the data from either dataset
    print('DS1') #indicates which dataset is being analyzed
    #reading the data using the csv python library
    trainDataRaw = csv.reader(open("Datasets/DS1-train.csv", "r"))
    validDataRaw = csv.reader(open("Datasets/DS1-valid.csv", "r"))
    testDataRaw = csv.reader(open("Datasets/DS1-test.csv", "r"))
else:
    print('DS2')
    trainDataRaw = csv.reader(open("Datasets/DS2-train.csv", "r"))
    validDataRaw = csv.reader(open("Datasets/DS2-valid.csv", "r"))
    testDataRaw = csv.reader(open("Datasets/DS2-test.csv", "r"))

trainData = []
validData = []
testData = []

#splitting the data and casting it to float to make it more usable, also removing the last, empty element
for row in trainDataRaw:
    trainData.append([float(i) for i in row[:21]])
for row in validDataRaw:
    validData.append([float(i) for i in row[:21]])
for row in testDataRaw:
    testData.append([float(i) for i in row[:21]])

#training the model on the training data by instantiating a object of the class
model = GDA(trainData)

print('Bias of the model: ' + str(model.bias))
print('Slope of the model: ' + str(model.slope))

print('Validation scores')
predictions = model.predict(validData)  #using the model to make predictions on the validation data
fp = 0
fn = 0
tp = 0
tn = 0
for i in range(len(predictions)):
    if predictions[i] == 1:
        if validData[i][-1] == 1:
            tp += 1 #numer of correctly classified class = 1 samples
        else:
            fp += 1 #number of incorrectly classified class = -1 samples
    else:
        if validData[i][-1] == 1:
            fn += 1 #number of incorrectly classified class = 1 samples
        else:
            tn += 1 #number of correctly classified class = -1 samples
#computing performance metrics needed to evaluate performance
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / len(predictions)
f1 = 2 * precision * recall / (precision + recall)
print('F1 score: ' + str(f1))
print('Accuracy: ' + str(accuracy))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))

print('Testing scores')
predictions = model.predict(testData)
fp = 0
fn = 0
tp = 0
tn = 0
for i in range(len(predictions)):
    if predictions[i] == 1:
        if testData[i][-1] == 1:
            tp += 1 #numer of correctly classified class = 1 samples
        else:
            fp += 1 #number of incorrectly classified class = -1 samples
    else:
        if testData[i][-1] == 1:
            fn += 1 #number of incorrectly classified class = 1 samples
        else:
            tn += 1 #number of correctly classified class = -1 samples
#computing performance metrics needed to evaluate performance
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / len(predictions)
f1 = 2 * precision * recall / (precision + recall)
print('F1 score: ' + str(f1))
print('Accuracy: ' + str(accuracy))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
