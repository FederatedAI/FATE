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
import copy
import io
import os
import tempfile
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from arch.api.utils import log_utils
from federatedml.framework.weights import OrderDictWeights, Weights
from federatedml.nn.homo_nn.nn_model import NNModel, DataConverter

Logger = log_utils.getLogger()


def _zip_dir_as_bytes(path):
    with io.BytesIO() as io_bytes:
        with zipfile.ZipFile(io_bytes, 'w', zipfile.ZIP_DEFLATED) as zipper:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    full_path = os.path.join(root, name)
                    relative_path = os.path.relpath(full_path, path)
                    zipper.write(filename=full_path, arcname=relative_path)
                for name in dirs:
                    full_path = os.path.join(root, name)
                    relative_path = os.path.relpath(full_path, path)
                    zipper.write(filename=full_path, arcname=relative_path)
        zip_bytes = io_bytes.getvalue()
    return zip_bytes


def _compile_model(model, loss, optimizer, metrics):
    optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)
    losses = getattr(tf.keras.losses, loss)
    model.compile(loss=losses,
                  optimizer=optimizer_instance,
                  metrics=metrics)
    return model


def _init_session():
    sess = tf.Session()
    tf.get_default_graph()
    set_session(sess)
    return sess


def _load_model(nn_struct_json):
    return tf.keras.models.model_from_json(nn_struct_json, custom_objects={})


def build_keras(nn_define, loss, optimizer, metrics, **kwargs):
    import json
    nn_define_json = json.dumps(nn_define)

    sess = _init_session()
    model = _load_model(nn_struct_json=nn_define_json)
    model = _compile_model(model=model,
                           loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
    return KerasNNModel(sess, model)


def from_keras_sequential_model(model, loss, optimizer, metrics):
    sess = _init_session()
    model = _compile_model(model=model,
                           loss=loss,
                           optimizer=optimizer,
                           metrics=metrics)
    return KerasNNModel(sess, model)


def restore_keras_nn_model(model_bytes):
    return KerasNNModel.restore_model(model_bytes)


class KerasNNModel(NNModel):
    def __init__(self, sess, model):
        self._sess: tf.Session = sess
        self._model: tf.keras.Sequential = model
        self._trainable_weights = {self._trim_device_str(v.name): v for v in self._model.trainable_weights}

    @staticmethod
    def _trim_device_str(name):
        return name.split("/")[0]

    def get_model_weights(self) -> OrderDictWeights:
        return OrderDictWeights(self._sess.run(self._trainable_weights))

    def set_model_weights(self, weights: Weights):
        unboxed = weights.unboxed
        self._sess.run([tf.assign(v, unboxed[name]) for name, v in self._trainable_weights.items()])

    def train(self, data: tf.keras.utils.Sequence, **kwargs):
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        self._model.fit(x=data, epochs=epochs, verbose=1, shuffle=True, **left_kwargs)
        return epochs * len(data)

    def evaluate(self, data: tf.keras.utils.Sequence, **kwargs):
        names = self._model.metrics_names
        values = self._model.evaluate(x=data, verbose=1)
        if not isinstance(values, list):
            values = [values]
        return dict(zip(names, values))

    def predict(self, data: tf.keras.utils.Sequence, **kwargs):
        return self._model.predict(data)

    def export_model(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            try:
                tf.keras.models.save_model(self._model, filepath=tmp_path, save_format="tf")
            except NotImplementedError:
                import warnings
                warnings.warn('Saving the model as SavedModel is still in experimental stages. '
                              'trying tf.keras.experimental.export_saved_model...')
                tf.keras.experimental.export_saved_model(self._model, saved_model_path=tmp_path)

            model_bytes = _zip_dir_as_bytes(tmp_path)

        return model_bytes

    @staticmethod
    def restore_model(model_bytes):  # todo: restore optimizer to support incremental learning
        sess = _init_session()
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as f:
                    f.extractall(tmp_path)

            try:
                model = tf.keras.models.load_model(filepath=tmp_path)
            except IOError:
                import warnings
                warnings.warn('loading the model as SavedModel is still in experimental stages. '
                              'trying tf.keras.experimental.load_from_saved_model...')
                model = tf.keras.experimental.load_from_saved_model(saved_model_path=tmp_path)
        return KerasNNModel(sess, model)

    def export_optimizer_config(self):
        return self._model.optimizer.get_config()


class KerasSequenceData(tf.keras.utils.Sequence):

    def get_shape(self):
        return self.x_shape, self.y_shape

    def __init__(self, data_instances, batch_size):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")

        _, one_data = data_instances.first()
        self.x_shape = one_data.features.shape

        num_label = len(data_instances.map(lambda x, y: [x, {y.label}]).reduce(lambda x, y: x | y))
        if num_label == 2:
            self.y_shape = (1,)
            self.x = np.zeros((self.size, *self.x_shape))
            self.y = np.zeros((self.size, *self.y_shape))
            index = 0
            self._keys = []
            for k, inst in data_instances.collect():
                self._keys.append(k)
                self.x[index] = inst.features
                self.y[index] = inst.label
                index += 1

        # encoding label in one-hot
        elif num_label > 2:
            self.y_shape = (num_label,)
            self.x = np.zeros((self.size, *self.x_shape))
            self.y = np.zeros((self.size, *self.y_shape))
            index = 0
            self._keys = []
            for k, inst in data_instances.collect():
                self._keys.append(k)
                self.x[index] = inst.features
                self.y[index][inst.label] = 1
                index += 1
        else:
            raise ValueError(f"num_label is {num_label}")

        self.batch_size = batch_size if batch_size > 0 else self.size

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return self.x[start: end], self.y[start: end]

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys


class KerasSequenceDataConverter(DataConverter):
    def convert(self, data, *args, **kwargs):
        return KerasSequenceData(data, *args, **kwargs)
