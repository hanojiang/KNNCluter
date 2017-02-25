# import csv
# import random
#
#
# def loadDataSet(fileName,split,trainingDataSet=[],targetDataSet=[]):
#     with open(fileName,'r') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         dataSet = list(csv_reader)
#         # print(dataSet)
#         print(len(dataSet))
#         for i in range(len(dataSet)):
#             for j in range(4):
#                 dataSet[i][j] = float(dataSet[i][j])
#                 # print(random.random() (0-1)之间
#             if random.random() < split:
#                     trainingDataSet.append(dataSet[i])
#             else:targetDataSet.append(dataSet[i])
#
#
# trainingDataSet = []
# targetDataSet = []
# loadDataSet(r'H:\文档\pythondata\KNNCluter\src\Iris.data.txt',0.6,trainingDataSet,targetDataSet)
# print(len(trainingDataSet))
# print(len(targetDataSet))



import csv
import random
import math
import operator


def loadDataSet(fileName,split,trainingDataSet,targetDataSet):
    with open(fileName,'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        dataSet = list(csv_reader)
        # print(dataSet)
        print(len(dataSet))
        for i in range(len(dataSet)):
            for j in range(4):
                dataSet[i][j] = float(dataSet[i][j])
                # print(random.random() (0-1)之间
            if random.random() < split:
                    trainingDataSet.append(dataSet[i])
            else:targetDataSet.append(dataSet[i])
        print(len(trainingDataSet))
        print(len(targetDataSet))

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
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
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.5
    loadDataSet(r'H:\文档\pythondata\KNNCluter\src\Iris.data.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)) )
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()