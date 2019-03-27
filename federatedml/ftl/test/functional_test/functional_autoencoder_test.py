#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

from federatedml.ftl.autoencoder import Autoencoder
from federatedml.ftl.test.util import assert_matrix


def getKaggleMNIST(file_path):

    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv(file_path)
    train = train.as_matrix()
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0].astype(np.int32)
    Xtest  = train[-1000:, 1:] / 255
    Ytest  = train[-1000:, 0].astype(np.int32)

    return Xtrain, Ytrain, Xtest, Ytest


def test_single_autoencoder():

    # To run this test, you may first download MINST dataset from kaggle:
    # https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer
    file_path = '../../../../data/MINST/train.csv'
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST(file_path)
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    _, D = Xtrain.shape

    autoencoder = Autoencoder(0)
    autoencoder.build(D, 200)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        autoencoder.set_session(session)
        session.run(init_op)
        autoencoder.fit(Xtrain, epoch=1, show_fig=True)

        i = np.random.choice(len(Xtest))
        x = Xtest[i]
        y = autoencoder.predict([x])

        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(y.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.show()

        model_parameters = autoencoder.get_model_parameters()

    # test whether autoencoder can be restored from stored model parameters
    tf.reset_default_graph()

    autoencoder_2 = Autoencoder(0)
    autoencoder_2.restore_model(model_parameters)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        autoencoder_2.set_session(session)
        session.run(init_op)

        y_hat = autoencoder_2.predict([x])

        plt.subplot(1, 2, 1)
        plt.imshow(y.reshape(28, 28), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(y_hat.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.show()

        assert_matrix(y, y_hat)


if __name__ == '__main__':
    test_single_autoencoder()

