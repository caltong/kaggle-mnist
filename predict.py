import numpy as np


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