import numpy as np
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
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

model_path = "model_nctu/"
mkdirs(model_path)

train_file = "train_nctu/" + "normal_train.csv"

data = np.loadtxt(train_file, delimiter=",").astype(np.float)

for i in range(len(data[0])):
    plt.figure(figsize=(12, 8))
    plt.plot(data[:, i])
    plt.show()

mean_list = np.mean(data, axis=0)
data = data - mean_list
print("mean_list:", mean_list)

maxarg_list = np.argmax(data, axis=0)
print("maxarg_list:", maxarg_list)

fs = 100
sec_before = 5
sec_after = 10

data = data[maxarg_list[-1] - sec_before * fs: maxarg_list[-1] + sec_after * fs]

std_list = np.std(data, axis=0)
data = data / std_list
print("std_list:", std_list)

temp = np.vstack((mean_list.reshape(1, -1), std_list.reshape(1, -1)))
np.savetxt("mean_std.txt", temp, delimiter=",")

look_back = 15
_data_X, _data_Y = create_dataset(data, look_back=look_back)

msk = np.random.rand(len(_data_X)) < 0.7
trainX = _data_X[msk]
valX = _data_X[~msk]

trainY = _data_Y[msk]
valY = _data_Y[~msk]

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(trainY.shape[1]))
model.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

history = model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2,
                    validation_data=[valX, valY], callbacks=[early_stopping])

model.save(model_path + 'model_lstm.h5')

predict = model.predict(_data_X, verbose=0)

test_list = []
for j in range(len(_data_Y[0])):
    testScore = (math.sqrt(mean_squared_error(_data_Y[:, j], predict[:, j])))
    testScore = testScore / std_list[j+1]
    test_list.append(testScore)
print("test_list:", test_list)

loss = history.history["loss"]
val_loss = history.history["val_loss"]

x = range(1, len(loss) + 1)
plt.figure(figsize=(12, 8))
plt.plot(x, loss, label='loss')
plt.plot(x, val_loss, label='val_loss')
plt.legend(loc='best', fontsize=15)
plt.title('Loss', fontsize=20)
plt.xlabel('Epoch')
plt.show()
