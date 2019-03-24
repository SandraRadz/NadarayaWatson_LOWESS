import math
import random

import pylab as pl
import scipy as sp
import numpy as np
from scipy._lib.six import reduce


def countY(x):
    y = []
    for xi in x:
        y.append((1 / (1 + 5 * xi ** 2)))
    return y


def countY2(x):
    y = []
    for xi in x:
        y.append(random.gauss((1 / (1 + 5 * xi ** 2)), 0.05))
    return y


def generateData(pointNum, minX, maxX, function):
    data = []
    x = sp.linspace(minX, maxX, pointNum)
    y = function(x)
    data = [[x[i], y[i]] for i in range(len(x))]
    return data


def showData(data, color):
    pl.scatter([data[i][0] for i in range(len(data))],
               [data[i][1] for i in range(len(data))],
               c=color)


def splitData(data):
    trainData = []
    testData = []
    for d in data:
        if random.random() < 0.7:
            trainData.append(d)
        else:
            testData.append(d)
    trainData.append([-10, 5])
    trainData.append([10, 6])
    trainData.append([8, 7])
    trainData.append([-7, 9])
    return trainData, testData


# result [[dist, [x, y]]]
def calcDistToPoint(point, trainData):
    calcDist = []
    for td in trainData:
        d = dist(point, td)
        calcDist.append([d, td])
    return calcDist


# a=x, b =[x, y]
def dist(a, b):
    return math.sqrt((a - b[0]) ** 2)


def rectangulaCore(r):
    if r<1:
        return 1
    return 0


def triangularCore(r):
    if r<1:
        return 1 - r
    return 0


def squareCore(r):
    if r<1:
        return (1 - r) ** 2
    return 0


def superSquareCore(r):
    if r<1:
        return (1 - (r) ** 2) ** 2
    return 0


def gaussCore(r):
    e = 2.71828
    if r:
        return e ** (-2 * (r) ** 2)
    return 0


def NadarayaWatson(x, h, trainData, coreType, q):
    distArr = calcDistToPoint(x, trainData)
    sum1 = 0
    sum2 = 0
    for i in range(len(distArr)):
        c = coreType(distArr[i][0] / h)*q[i]
        sum1 += distArr[i][1][1] * c
        sum2 += c
    if sum2 == 0:
        return 0
    res = sum1 / sum2
    return res


def error(data):
    sum = 0
    for d in data:
        sum += math.sqrt((d[2] - d[1]) ** 2)
    return sum


def trainModelH(trainData, realRes, hfinish, step, core):
    q =[1 for i in range(len(trainData))]
    variants = []
    h = step
    while h <= hfinish:
        errClacArr = []
        for rr in realRes:
            myY = NadarayaWatson(rr[0], h, trainData, core, q)
            errClacArr.append([rr[0], rr[1], myY])
        variants.append([error(errClacArr), h])
        h += step
    minErr = variants[0][0]
    minH = variants[0][1]
    for v in variants:
        if v[0] < minErr:
            minErr = v[0]
            minH = v[1]
    return minH, minErr


def trainModelCoreType(trainData, realRes, hfinish, step, coreArr):
    variants = []
    for ct in coreArr:
        h, err = trainModelH(trainData, realRes, hfinish, step, ct)
        variants.append([err, h, ct])
    minErr = variants[0][0]
    minH = variants[0][1]
    core = variants[0][2]
    for v in variants:
        if v[0] < minErr:
            minErr = v[0]
            minH = v[1]
            core = v[2]
    return minH, minErr, core


def interpolate(x, y, xq):
    def _basis(j):
        p = [(xq - x[i]) / (x[j] - x[i]) for i in range(k) if i != j]
        return reduce((lambda q, w: q * w), p)

    assert len(x) != 0 and (
            len(x) == len(y)), 'x and y cannot be empty and must have the same length'
    k = len(x)
    return sum(_basis(j) * y[j] for j in range(k))


def lowess(trainData, h, coreType):
    q = [1 for i in trainData]
    for iter in range(4):
        a = []
        for point in trainData:
            dataWithoutPoint = []
            for td in trainData:
                if td[0] != point[0]:
                    dataWithoutPoint.append(td)
            ch = 0
            z = 0
            for i in range(len(dataWithoutPoint)):
                c = q[i] * coreType(dist(point[0], dataWithoutPoint[i]) / h)
                ch += dataWithoutPoint[i][1] * c
                z += c
            a.append(ch / z)
        for j in range(len(q)):
            q[j] = coreType(math.fabs(a[j] - trainData[j][1]))
    return q



np.seterr(divide='ignore', invalid='ignore')
# generateData
data = generateData(70, -15, 15, countY2)
trainData, testData = splitData(data)

coreTypes = [rectangulaCore, triangularCore, squareCore, superSquareCore, gaussCore]
trainModel = []
# for ct in coreTypes:
#    trainModel.append([])
minH, minErr, ct = trainModelCoreType(trainData, testData, 5, 0.5, coreTypes)
print(minH)
print(minErr)
print(ct)
# create results
result = []
q = lowess(trainData, minH, ct)
for i in testData:
    result.append([i[0], NadarayaWatson(i[0], minH, trainData, ct, q)])


# show results
showData(trainData, "red")
showData(testData, "blue")
showData(result, "green")
pl.show()
