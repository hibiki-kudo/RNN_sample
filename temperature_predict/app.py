import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from rnn import RnnLSTM
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan


def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)

rnn = RnnLSTM("temperature.h5")

def data_set_read(during):
    data_set_path = "./temperature_2000_2018/"
    datas = pd.read_csv(data_set_path + during + ".csv")
    return datas


def data_arrange_one_year(data):
    days_data = []

    for temperature in data.as_matrix():
        days_data.append(temperature)

        if "/12/31" in temperature[0]:
            yield days_data
            days_data = []


def generate_data(data, length_data, dimension):
    sequences = []
    # 正解データを入れる箱
    target = []
    # 正解データの日付を入れる箱
    # target_date = []

    # 一グループごとに時系列データと正解データをセットしていく
    for i in range(len(data) - length_data):
        try:
            sequences.append(data[i:i + length_data])
            target.append(data[i + length_data][1])
            # target_date.append(data[i + length_data][0])
        except KeyError:
            target.append(data[0][1])
            # target_date.append(data[0][0])

    # 時系列データを成形
    X = np.array(sequences).reshape(len(sequences), length_data, dimension)
    # 正解データを成形
    Y = np.array(target).reshape(len(sequences), 1)
    # 正解データの日付データを成形
    # Y_date = np.array(target_date).reshape(len(sequences), 1)
    # print(X)

    # return (X, Y, Y_date)
    return (X, Y)


def predict_result_display(X_test, Y_test):
    global rnn
    predicted = []
    for i in range(0, len(X_test)):
        y_ = rnn.predict(X_test[i:i + 1, :, :])
        predicted.append(y_)

    plt.plot(predicted, color='black')
    plt.plot(Y_test)
    plt.show()

def main():
    global rnn
    durings = ["1977_1982","1982_1988","1988_1994", "1994_2000","2000_2006", "2006_2012", "2012_2018"]

    length_data = 366  # データのサイズ
    dimension = 5  # データの要素数
    rnn.create_model(length_data, dimension, 100)
    # rnn.save_model()
    # rnn.load_model()

    # for a_year_data in list(data_arrange_one_year(data)):
    # for values in data.as_matrix():
    #     print(len(values[0]))
    #     X_train, Y_train, Y_dates = generate_data(values, length_data, dimension)
    #     rnn.train(X_train=X_train, Y_train=Y_train, epochs=10, batch_size=10)
    #     rnn.save_model()

    for during in durings:
        print(during)
        data = data_set_read(during)

        X_train, Y_train = generate_data(data.as_matrix(['最高気温(℃)', '最低気温(℃)', '平均気温(℃)', '降水量の合計(mm)', '日照時間(時間)']), length_data, dimension)
        rnn.train(X_train=X_train, Y_train=Y_train, epochs=10, batch_size=10)
        rnn.save_model()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--debug':
            set_debugger_session()
        else:
            raise ValueError('unkown option :{}'.format(sys.argv[1]))
    main()
