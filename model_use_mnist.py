#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import struct
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 10
epochs = 32
img_rows, img_cols = 28, 28


def load_kaggle_data(path):
    temp_list = []
    with open(path) as file:
        lines = csv.reader(file)
        for line in lines:
            temp_list.append(line)  # 第一列为标签 第一行为数据名称
    temp_list.remove(temp_list[0])  # 去除第一行
    # toInt
    # for i in range(len(l)):
    #     for j in range(len(l[0])):
    #         l[i][j] = int(l[i][j])
    temp_list = np.array(temp_list)
    temp_list = temp_list.astype(np.int)
    label = temp_list[:, 0]
    data = temp_list[:, 1:]
    data = data.astype(np.float32)
    data = data / 255
    data = data.reshape(data.shape[0], 28, 28, 1)
    return data, label


def load_mnist_data(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 28, 28, 1])  # reshape为[60000,784]型数组

    return imgs, head


def load_mnist_label(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head


x_train_kaggle, y_train_kaggle = load_kaggle_data('train.csv')
x_train_mnist, _ = load_mnist_data('train-images.idx3-ubyte')
y_train_mnist,_ = load_mnist_label('train-labels.idx1-ubyte')


x_train = np.append(x_train_kaggle, x_train_mnist, axis=0)
y_train = np.append(y_train_kaggle,y_train_mnist)
print(x_train_kaggle.shape)
print(x_train_mnist.shape)
print(x_train.shape)
print(y_train.shape)


# model = keras.models.load_model('model.h5')
