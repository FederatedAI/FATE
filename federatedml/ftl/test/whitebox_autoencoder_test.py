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

import unittest

import numpy as np
import tensorflow as tf

from federatedml.ftl.autoencoder import Autoencoder
from federatedml.ftl.test.util import assert_matrix


class TestAutoencoder(unittest.TestCase):

    def test_autoencoder_restore_model(self):

        X = np.array([[4, 2, 3],
                      [6, 5, 1],
                      [3, 4, 1],
                      [1, 2, 3]])

        _, D = X.shape

        tf.reset_default_graph()
        autoencoder = Autoencoder(0)
        autoencoder.build(D, 5)
        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            autoencoder.set_session(session)
            session.run(init_op)
            autoencoder.fit(X, epoch=10)
            model_parameters = autoencoder.get_model_parameters()

        tf.reset_default_graph()

        autoencoder.restore_model(model_parameters)
        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            autoencoder.set_session(session)
            session.run(init_op)
            Wh = autoencoder.Wh.eval()
            Wo = autoencoder.Wo.eval()
            bh = autoencoder.bh.eval()
            bo = autoencoder.bo.eval()

        assert_matrix(model_parameters["Wh"], Wh)
        assert_matrix(model_parameters["Wo"], Wo)
        assert_matrix(model_parameters["bh"], bh)
        assert_matrix(model_parameters["bo"], bo)


if __name__ == '__main__':
    unittest.main()