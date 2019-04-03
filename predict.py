import numpy as np
import csv
import keras
import pandas as pd


def load_test_data(path):
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
    # label = temp_list[:, 0]
    data = temp_list
    data = data.astype(np.float32)
    data = data / 255.0
    return data


img_rows, img_cols = 28, 28

x_predict = load_test_data('test.csv')
x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
print(x_predict.shape)

model = keras.models.load_model('model_use_mnist.h5')
y_predict = model.predict_classes(x_predict, verbose=1)
pd.DataFrame({"ImageId": list(range(1, len(y_predict) + 1)), "Label": y_predict}).to_csv('submission_use_mnist.csv',
                                                                                         index=False, header=True)
# print(y_predict.shape, y_predict[99][1], y_predict[99][2])
