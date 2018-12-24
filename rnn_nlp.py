# -*- coding: utf-8 -*-
from keras.layers import Dense, Activation, SimpleRNN
from keras.models import Sequential, load_model
from keras.backend import clear_session
import tensorflow as tf
import numpy as np
import codecs

input_file = "alice_in_wonderland.txt"

with codecs.open(input_file, "r", encoding="utf_8") as f:
    lines = [line.strip().lower() for line in f if len(line) != 0]
    text = " ".join(lines)

chars = set(text)
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []

for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)

for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1

    y[i, char2index[label_chars[i]]] = 1


def RNN_model_create():
    global model
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=False, input_shape=(SEQLEN, nb_chars), unroll=True))
    model.add(Dense(nb_chars))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


def learn_RNN(iteration):
    # model = load_model("rnn_sample_alice.h5")
    print("=" * 50)
    print("Iteration #: {}".format(iteration))
    model.fit(X, y, epochs=1, batch_size=128, verbose=1)
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]

    print("Generating from seed: {}".format(test_chars))
    print(test_chars, end="")

    for i in range(100):
        Xtest = np.zeros((1, SEQLEN, nb_chars))

        for j, ch in enumerate(test_chars):
            Xtest[0, j, char2index[ch]] = 1

        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        test_chars = test_chars[1:] + ypred
    print()
    model.save("rnn_sample_alice.h5")

if __name__ == "__main__":
    # RNN_model_create()
    # for i in range(20):
    for i in range(10):
        learn_RNN(i)
    # RNN_predict()
