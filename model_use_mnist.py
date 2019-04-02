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


def load_mnist_data(file_name):
    #   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。
    #   file object = open(file_name [, access_mode][, buffering])
    #   file_name是包含您要访问的文件名的字符串值。
    #   access_mode指定该文件已被打开，即读，写，追加等方式。
    #   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。
    #   这里rb表示只能以二进制读取的方式打开一个文件
    with open(file_name, 'rb') as bin_file:
        #   从一个打开的文件读取数据
        buffers = bin_file.read()
        #   读取image文件前4个整型数字
        magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
        #   整个images数据大小为60000*28*28
        bits = num * rows * cols
        #   读取images数据
        images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
        #   转换为[60000,784]型数组
        # images = np.reshape(images, [num, rows * cols])
        images = np.array(images)
        images = images.reshape(num, rows, cols, 1)
    return images


x_train_kaggle, y_train_kaggle = load_kaggle_data('train.csv')
x_train_mnist = load_mnist_data('train-images.idx3-ubyte')
x_train = np.append(x_train_kaggle,x_train_mnist,axis=0)
print(x_train_kaggle.shape)
print(x_train_mnist.shape)
print(x_train.shape)

# model = keras.models.load_model('model.h5')
