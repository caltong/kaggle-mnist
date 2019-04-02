import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 128
num_classes = 10
epochs = 12
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
train_data = train_data.T
train_label = train_label.reshape(1, -1)
print(train_data.shape, train_label.shape)
print(train_label[0][1])

x_train = train_data[:40000]
y_train = train_label[:40000]
x_test = train_data[40000:]
y_test = train_label[40000:]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
