import csv
import numpy
import re

class NN:
    def __init__(self, data):       #simply stores the data, NN has no training stage
        self.data = data

    def predict(self, k, x):
        results = []
        #cycles through the data to be predicted
        for givenData in x:
            #sets the error to an impossible value to be replaced later
            bestError = [-1 for i in range(k)]
            #sets the class to a value of 0 so that if it ends up in the prediction (k>#data) then it doesn't affect the prediction
            predictedClass = [0 for i in range(k)]
            #cycling through the training data
            for sample in self.data:
                sumError = 0
                #computing the manhattan distance between the sample to be predicted and the training sample
                for featIndex in range(len(sample[:-1])):
                    sumError += abs(givenData[featIndex] - sample[featIndex])
                #inserts the sample in the list if the error is less than any of the previously found errors
                for errorIndex in range(len(bestError)):
                    if sumError < bestError[errorIndex] or bestError[errorIndex] < 0:
                        bestError.insert(errorIndex, sumError)
                        bestError.pop()
                        predictedClass.insert(errorIndex, sample[-1])
                        predictedClass.pop()
                        break
            #sums the classes to do majority voting, since the classes are 1 and -1
            #tiebreaker = pick class -1
            if sum(predictedClass) > 0:
                results.append(1)
            else:
                results.append(-1)
        return results

if False:   #switch for reading the data from either dataset
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
model = NN(trainData)
results = []

print('Validation scores')
for k in range(1,10):
    predictions = model.predict(k, validData) #using the model to make predictions on the validation data
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
    f1 = 2 * precision * recall / (precision + recall)
    print('F1 score for k = ' + str(k) + ': ' + str(f1))
    results.append(f1)

#this loop finds the best value of k using the F1 score
bestScore = results[0]
bestIndex = 0
for i in range(len(results[1:])):
    if results[i] > bestScore:
        bestScore = results[i]
        bestIndex = i


print('BEST FIT (k = ' + str(bestIndex + 1) + '): ')

#computes the predictions based on this best value of k found
predictions = model.predict(bestIndex + 1, testData)
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
