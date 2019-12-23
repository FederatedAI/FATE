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
import numpy as np
import tensorflow as tf

from federatedml.framework.weights import OrderDictWeights
from federatedml.nn.homo_nn.nn_model import NNModel, DataConverter


class TFNNModel(NNModel):
    def __init__(self, optimizer, loss, metrics, predict_ops):
        self._optimizer = optimizer
        self._loss = loss

        if not isinstance(metrics, list):
            metrics = [metrics]
        self._metrics = {TFNNModel._trim_device_str(metric.name): metric for metric in metrics}

        self._train_op = self._optimizer.minimize(self._loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self._trainable_weights = {self._trim_device_str(v.name): v for v in
                                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)}

        self._predict_ops = predict_ops

    @staticmethod
    def _trim_device_str(name):
        return name.split("/")[0]

    def get_model_weights(self):
        return OrderDictWeights(self._sess.run(self._trainable_weights))

    def set_model_weights(self, weights):
        unboxed = weights.unboxed
        self._sess.run([tf.assign(v, unboxed[name]) for name, v in self._trainable_weights.items()])

    def get_batch_evaluate(self, batch):
        return self._sess.run(self._metrics, feed_dict=batch)

    def train_batch(self, batch):
        return self._sess.run(self._train_op, feed_dict=batch)

    def train(self, data, **kwargs):
        for batch in data:
            self.train_batch(batch)

    def evaluate(self, data, **kwargs):
        total = {name: 0.0 for name in self._metrics}
        for batch in data:
            for k, v in self.get_batch_evaluate(batch).item():
                total[k] += v
        return total

    def export_model(self):
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            self._sess,
            self._sess.graph_def,
            [v.name.split(":")[0] for v in self._predict_ops.values()]
        )
        return frozen_graph_def.SerializeToString()


class TFFitDictData(object):

    def __init__(self, data_instances, batch_size, **kwargs):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")

        one_data = data_instances.first()
        x_shape = one_data[1][0].shape
        y_shape = one_data[1][1].shape
        self.x = np.zeros((self.size, *x_shape))
        self.y = np.zeros((self.size, *y_shape))
        index = 0
        for k, v in data_instances.collect():
            self.x[index] = v[0]
            self.y[index] = v[1]
            index += 1
        self.batch_size = batch_size
        self.additional_kv = kwargs

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return dict(x=self.x[start: end], y=self.y[start: end], **self.additional_kv)

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def __iter__(self):
        """Creates an infinite generator that iterate over the Sequence.

        Yields:
          Sequence items.
        """
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item


class TFFitDictDataConverter(DataConverter):
    def convert(self, data, *args, **kwargs):
        return TFFitDictData(data, *args, **kwargs)

