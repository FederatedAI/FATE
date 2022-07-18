import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential

regressor = Sequential()

inputs = tf.random.normal([32, 10, 8])
lstm = LSTM(4, return_sequences=True)
regressor.add(lstm)
regressor.add(Dense(units=1))
output = regressor(inputs)

import torch as t
import numpy as np

layer = t.nn.Linear(10, 3, True)
a = t.tensor(np.random.random((100, 20, 10)), dtype=t.float32)