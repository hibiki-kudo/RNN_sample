from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import tensorflow as tf
import sys

class RnnLSTM:

    def __init__(self, file_name=""):
        self.model = None
        self.save_model_file = file_name

    def create_model(self, length_of_sequence, dimension, n_hidden):
        # 1つの学習データのStep数(今回は25)
        # length_of_sequence = g.shape[1]
        # in_out_neurons = 1
        # n_hidden = 300

        self.model = Sequential()
        self.model.add(
            LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, dimension), return_sequences=False))
        self.model.add(Dense(1))
        self.model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        self.model.compile(loss="mean_squared_logarithmic_error", optimizer=optimizer, metrics=["accuracy"])
        # モデルの要約を出力
        self.model.summary()

    def train(self, X_train, Y_train, epochs, batch_size):
        # try:

        model_ckp = ModelCheckpoint(filepath=self.save_model_file, monitor='loss', verbose=1, save_best_only=True, mode='auto',
                                    period=5)
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[model_ckp])
        # except:
        #     print("モデル生成できていません")

    def predict(self, X):
        try:
            return self.model.predict(X, verbose=1)
        except:
            print("モデル生成できていません")

    def save_model(self):
        self.model.save(self.save_model_file)

    def load_model(self):
        self.model = load_model(self.save_model_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clear':
            tf.keras.backend.clear_session()

