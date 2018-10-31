import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
import math
import operator


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors




def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


(X_train, y_train), (X_test, y_test) = mnist.load_data()

trained_images = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
test_images = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

trainSet = trained_images[:100]
trained_labels = y_train[:100]

testSet = test_images[:100]
test_labels = y_test[:100]
testInstance = testSet[0]
distance = euclideanDistance(trainSet[10], testInstance, 784)
print('Distance: ' + repr(distance))
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)

response = getResponse(neighbors)
print(response)
# x=iris.data
# y=iris.target
# x_train=x[:100]
# y_train=y[:100]
#
# x_validate=x[:100]
# y_validate=y[:100]

# neighbor=KNeighborsClassifier(n_neighbors=5,weights='uniform')
# neighbor.fit(x_train,y_train)
# predicted=neighbor.predict(x_test)
#
# print(("predicetd=",predicted))
# print(("predicetd=",y_test))
