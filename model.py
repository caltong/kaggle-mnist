import numpy as np
import tensorflow as tf
import csv


def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    # toInt
    # for i in range(len(l)):
    #     for j in range(len(l[0])):
    #         l[i][j] = int(l[i][j])
    l = np.array(l)
    l = l.astype(np.float32)
    l = l / 255
    label = l[:, 0]
    data = l[:, 1:]
    return label, data

# todo def loadTestData():

train_X, train_Y = loadTrainData()
# train_X = train_Y.T
# train_Y = train_Y.T
print(train_X.shape, train_Y.shape)