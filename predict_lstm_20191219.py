import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import math
from sklearn.metrics import mean_squared_error

np.random.seed(87)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 創目錄
def mkdirs(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=15, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - time_step - 1):
        a = np.hstack((dataset[i + 1:(i + look_back + 1), 0].reshape(-1, 1), dataset[i:(i + look_back), 1:]))
        dataX.append(a)
        dataY.append(dataset[i + look_back + time_step - 1, 1:])
    return np.array(dataX), np.array(dataY)


model_name = "model_nctu/model_lstm.h5"
model = load_model(model_name)

test_files = ["test_nctu/" + "normal_test.csv",
              "test_nctu/" + "abnormal_test.csv"]

mean_std = np.loadtxt("mean_std.txt", delimiter=",")
mean_list = mean_std[0]
std_list = mean_std[1]

fs = 100
sec_before = 5
sec_after = 10
look_back = 15

for test_file in test_files:
    data = np.loadtxt(test_file, delimiter=",").astype(np.float)

    for i in range(len(data[0])):
        plt.figure(figsize=(12, 8))
        plt.plot(data[:, i])
        plt.show()

    data = data - mean_list

    maxarg_list = np.argmax(data, axis=0)
    print("maxarg_list:", maxarg_list)
    data = data[maxarg_list[-1] - sec_before * fs: maxarg_list[-1] + sec_after * fs]

    std_list_2 = np.std(data, axis=0)

    data = data / std_list

    _data_X, _data_Y = create_dataset(data, look_back=look_back)

    predict = model.predict(_data_X, verbose=0)

    test_list = []
    for j in range(len(_data_Y[0])):
        testScore = (math.sqrt(mean_squared_error(_data_Y[:, j], predict[:, j])))
        testScore = testScore / std_list_2[j + 1]
        test_list.append(testScore)
    print("test_list:", test_list)

    floor = [1, 2, 4, 7, 8]
    x = np.arange(0, len(_data_Y)) / fs
    for i in range(len(_data_Y[0])):
        plt.figure(figsize=(12, 16))
        plt.subplot(211)
        plt.title("Floor_" + str(floor[i]), fontsize=40)
        plt.plot(x, _data_Y[:, i] * std_list[i+1], label='observation')
        plt.plot(x, predict[:, i] * std_list[i+1], label='prediction')
        plt.legend(loc='best', fontsize=15)
        plt.ylabel('amplitude(G)', fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.margins(0.08)
        plt.subplots_adjust(bottom=0.15)

        plt.subplot(212)
        plt.title("Difference", fontsize=40)
        plt.plot(x, _data_Y[:, i] * std_list[i+1] - predict[:, i] * std_list[i+1])
        plt.ylabel('amplitude(G)', fontsize=30)  #
        plt.xlabel('time(S)', fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.margins(0.08)
        plt.subplots_adjust(bottom=0.15)
        plt.show()