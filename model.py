# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 10
epochs = 32
img_rows, img_cols = 28, 28


def load_train_data(path):
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
    return data, label


train_data, train_label = load_train_data('train.csv')
train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
print(train_data.shape, train_label.shape)

x_train = train_data[:40000]
y_train = train_label[:40000]
x_test = train_data[40000:]
y_test = train_label[40000:]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save('model.h5')
print('Test loss:', score[0])
print('Test accuracy', score[1])
