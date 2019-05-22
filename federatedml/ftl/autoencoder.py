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
import tensorflow as tf

from federatedml.ftl.eggroll_computation.helper import distribute_compute_sum_XY


class Autoencoder(object):

    def __init__(self, an_id):
        super(Autoencoder, self).__init__()
        self.id = str(an_id)
        self.sess = None
        self.built = False
        self.lr = None
        self.input_dim = None
        self.hidden_dim = None

    def set_session(self, sess):
        self.sess = sess

    def get_session(self):
        return self.sess

    def build(self, input_dim, hidden_dim, learning_rate=1e-2):
        if self.built:
            return

        self.lr = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._set_variable_initializer()
        self._build_model()

    def _set_variable_initializer(self):
        self.Wh_initializer = tf.random_normal((self.input_dim, self.hidden_dim), dtype=tf.float64)
        self.bh_initializer = np.zeros(self.hidden_dim).astype(np.float64)
        self.Wo_initializer = tf.random_normal((self.hidden_dim, self.input_dim), dtype=tf.float64)
        self.bo_initializer = np.zeros(self.input_dim).astype(np.float64)

    def _build_model(self):
        self._add_input_placeholder()
        self._add_encoder_decoder_ops()
        self._add_forward_ops()
        self._add_representation_training_ops()
        self._add_e2e_training_ops()
        self._add_encrypt_grad_update_ops()
        self.built = True

    def _add_input_placeholder(self):
        self.X_in = tf.placeholder(tf.float64, shape=(None, self.input_dim))

    def _add_encoder_decoder_ops(self):
        self.encoder_vars_scope = self.id + "_encoder_vars"
        with tf.variable_scope(self.encoder_vars_scope):
            self.Wh = tf.get_variable("weights", initializer=self.Wh_initializer, dtype=tf.float64)
            self.bh = tf.get_variable("bias", initializer=self.bh_initializer, dtype=tf.float64)

        self.decoder_vars_scope = self.id + "_decoder_vars"
        with tf.variable_scope(self.decoder_vars_scope):
            self.Wo = tf.get_variable("weights", initializer=self.Wo_initializer, dtype=tf.float64)
            self.bo = tf.get_variable("bias", initializer=self.bo_initializer, dtype=tf.float64)

    def _add_forward_ops(self):
        self.Z = self._forward_hidden(self.X_in)
        self.logits = self._forward_logits(self.X_in)
        self.X_hat = self._forward_output(self.X_in)

    def _add_representation_training_ops(self):
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_vars_scope)
        self.init_grad = tf.placeholder(tf.float64, shape=(None, self.hidden_dim))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.Z, var_list=vars_to_train,
                                                                               grad_loss=self.init_grad)

    def _add_e2e_training_ops(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.X_in))
        self.e2e_train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _add_encrypt_grad_update_ops(self):
        self.Z_grads = tf.gradients(self.Z, xs=[self.Wh, self.bh])

        self.grads_W_new = tf.placeholder(tf.float64, shape=[self.input_dim, self.hidden_dim])
        self.grads_b_new = tf.placeholder(tf.float64, shape=[self.hidden_dim])
        self.new_Wh = self.Wh.assign(self.Wh - self.lr * self.grads_W_new)
        self.new_bh = self.bh.assign(self.bh - self.lr * self.grads_b_new)

    def _forward_hidden(self, X):
        return tf.sigmoid(tf.matmul(X, self.Wh) + self.bh)

    def _forward_logits(self, X):
        Z = self._forward_hidden(X)
        return tf.matmul(Z, self.Wo) + self.bo

    def _forward_output(self, X):
        return tf.sigmoid(self._forward_logits(X))

    def transform(self, X):
        return self.sess.run(self.Z, feed_dict={self.X_in: X})

    def compute_gradients(self, X):
        grads_W_collector = []
        grads_b_collector = []
        for i in range(len(X)):
            grads_w_i, grads_b_i = self.sess.run(self.Z_grads, feed_dict={self.X_in: np.expand_dims(X[i], axis=0)})
            grads_W_collector.append(grads_w_i)
            grads_b_collector.append(grads_b_i)
        return [np.array(grads_W_collector), np.array(grads_b_collector)]

    def compute_encrypted_params_grads(self, X, encrypt_grads):
        grads = self.compute_gradients(X)
        grads_W = grads[0]
        grads_b = grads[1]
        encrypt_grads_ex = np.expand_dims(encrypt_grads, axis=1)
        encrypt_grads_W = distribute_compute_sum_XY(encrypt_grads_ex, grads_W)
        encrypt_grads_b = distribute_compute_sum_XY(encrypt_grads, grads_b)
        return encrypt_grads_W, encrypt_grads_b

    def apply_gradients(self, gradients):
        grads_W = gradients[0]
        grads_b = gradients[1]
        _, _ = self.sess.run([self.new_Wh, self.new_bh],
                             feed_dict={self.grads_W_new: grads_W, self.grads_b_new: grads_b})

    def backpropogate(self, X, y, in_grad):
        self.sess.run(self.train_op, feed_dict={self.X_in: X, self.init_grad: in_grad})

    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X_in: X})

    def get_encode_dim(self):
        return self.hidden_dim

    def get_model_parameters(self):
        _Wh = self.sess.run(self.Wh)
        _Wo = self.sess.run(self.Wo)
        _bh = self.sess.run(self.bh)
        _bo = self.sess.run(self.bo)

        hyperparameters = {"learning_rate": self.lr,
                           "input_dim": self.input_dim,
                           "hidden_dim": self.hidden_dim}
        return {"Wh": _Wh, "bh": _bh, "Wo": _Wo, "bo": _bo, "hyperparameters": hyperparameters}

    def restore_model(self, model_parameters):
        self.Wh_initializer = model_parameters["Wh"]
        self.bh_initializer = model_parameters["bh"]
        self.Wo_initializer = model_parameters["Wo"]
        self.bo_initializer = model_parameters["bo"]
        model_meta = model_parameters["hyperparameters"]

        self.lr = model_meta["learning_rate"]
        self.input_dim = model_meta["input_dim"]
        self.hidden_dim = model_meta["hidden_dim"]
        self._build_model()

    def fit(self, X, batch_size=32, epoch=1, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_size
        costs = []
        for ep in range(epoch):
            for i in range(n_batches + 1):
                batch = X[i * batch_size: i * batch_size + batch_size]
                _, c = self.sess.run([self.e2e_train_op, self.loss], feed_dict={self.X_in: batch})
                if i % 5 == 0:
                    print(i, "/", n_batches, "cost:", c)
                costs.append(c)

        if show_fig:
            plt.plot(costs)
            plt.show()
