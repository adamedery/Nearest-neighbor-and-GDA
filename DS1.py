import csv
import numpy
import re
from random import shuffle

#reading the data files using the python library for csv files
file_mean_0 = csv.reader(open("Datasets/DS1_m_0.csv", "r"))
file_mean_1 = csv.reader(open("Datasets/DS1_m_1.csv", "r"))
file_cov = csv.reader(open("Datasets/DS1_Cov.csv", "r"))

#splitting the metadata and casting it to float to make it more usable, also removing the last, empty element
for row in file_mean_0:
    mean0 = [float(i) for i in row[:20]]
for row in file_mean_1:
    mean1 = [float(i) for i in row[:20]]

#doing the same thing but here it must be done using arrays as there is more than one row per file
covariance = []
for row in file_cov:
    covariance.append([float(i) for i in row[:20]])

#setting the size of the data to be generated and the markers at which to switch from train to validation to test
data_size = 2000
marker1 = 1200
marker2 = 1600

#generating the data fro the 2 classes using the means provided for each feature and the covariance matrix
negativeData = numpy.random.multivariate_normal(mean0,covariance,size=data_size)
positiveData = numpy.random.multivariate_normal(mean1,covariance,size=data_size)

#putting the data into the format I want to save it in, csv, and appending the proper class to the data
negativeDataStrings = []
for sample in negativeData:
    negativeDataStrings.append(','.join(map(str, sample)) + ',-1\n')
positiveDataStrings = []
for sample in positiveData:
    positiveDataStrings.append(','.join(map(str, sample)) + ',1\n')

#mixing the data from both classes into train, validation and test sets based on the markers set earlier
trainData = negativeDataStrings[:marker1] + positiveDataStrings[:marker1]
validData = negativeDataStrings[marker1:marker2] + positiveDataStrings[marker1:marker2]
testData = negativeDataStrings[marker2:] + positiveDataStrings[marker2:]

#shuffling the data to ensure a random distribution of the classes
shuffle(trainData)
shuffle(validData)
shuffle(testData)

#outputting the data to the correct files line by line
with open("Datasets/DS1-train.csv", "w") as file:
    for sample in trainData:
        file.write(sample)

with open("Datasets/DS1-valid.csv", "w") as file:
    for sample in validData:
        file.write(sample)

with open("Datasets/DS1-test.csv", "w") as file:
    for sample in testData:
        file.write(sample)
