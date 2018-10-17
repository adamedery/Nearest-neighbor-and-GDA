import csv
import numpy
import re
from random import shuffle

#reading the data files using the python library for csv files
file_c1_m1 = csv.reader(open("Datasets/DS2_c1_m1.csv", "r"))
file_c1_m2 = csv.reader(open("Datasets/DS2_c1_m2.csv", "r"))
file_c1_m3 = csv.reader(open("Datasets/DS2_c1_m3.csv", "r"))
file_c2_m1 = csv.reader(open("Datasets/DS2_c2_m1.csv", "r"))
file_c2_m2 = csv.reader(open("Datasets/DS2_c2_m2.csv", "r"))
file_c2_m3 = csv.reader(open("Datasets/DS2_c2_m3.csv", "r"))
file_cov1 = csv.reader(open("Datasets/DS2_Cov1.csv", "r"))
file_cov2 = csv.reader(open("Datasets/DS2_Cov2.csv", "r"))
file_cov3 = csv.reader(open("Datasets/DS2_Cov3.csv", "r"))

#splitting the metadata and casting it to float to make it more usable, also removing the last, empty element
for row in file_c1_m1:
    c1_m1 = [float(i) for i in row[:20]]
for row in file_c1_m2:
    c1_m2 = [float(i) for i in row[:20]]
for row in file_c1_m3:
    c1_m3 = [float(i) for i in row[:20]]
for row in file_c2_m1:
    c2_m1 = [float(i) for i in row[:20]]
for row in file_c2_m2:
    c2_m2 = [float(i) for i in row[:20]]
for row in file_c2_m3:
    c2_m3 = [float(i) for i in row[:20]]

#doing the same thing but here it must be done using arrays as there is more than one row per file
cov1 = []
cov2 = []
cov3 = []
for row in file_cov1:
    cov1.append([float(i) for i in row[:20]])
for row in file_cov2:
    cov2.append([float(i) for i in row[:20]])
for row in file_cov3:
    cov3.append([float(i) for i in row[:20]])

#setting the size of the data to be generated and the markers at which to switch from train to validation to test
data_size = 2000
marker1 = 1200
marker2 = 1600

#generating the data fro the 2 classes using the means provided for each feature and the covariance matrix
#the if statements assert that the probability distribution for the different gaussians is followed
negativeData = []
for i in numpy.random.rand(2000):
    if i < 0.1:
        negativeData.append(numpy.random.multivariate_normal(c1_m1,cov1))
    elif i < 0.52:
        negativeData.append(numpy.random.multivariate_normal(c1_m2,cov2))
    else:
        negativeData.append(numpy.random.multivariate_normal(c1_m3,cov3))

positiveData = []
for i in numpy.random.rand(2000):
    if i < 0.1:
        positiveData.append(numpy.random.multivariate_normal(c2_m1,cov1))
    elif i < 0.52:
        positiveData.append(numpy.random.multivariate_normal(c2_m2,cov2))
    else:
        positiveData.append(numpy.random.multivariate_normal(c2_m3,cov3))

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
with open("Datasets/DS2-train.csv", "w") as file:
    for sample in trainData:
        file.write(sample)

with open("Datasets/DS2-valid.csv", "w") as file:
    for sample in validData:
        file.write(sample)

with open("Datasets/DS2-test.csv", "w") as file:
    for sample in testData:
        file.write(sample)
