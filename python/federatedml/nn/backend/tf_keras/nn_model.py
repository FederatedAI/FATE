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
import uuid

import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import gradients
from tensorflow.keras.callbacks import History
from tensorflow.python.keras import backend

try:
    from tensorflow import (
        get_default_graph,
        global_variables,
        initialize_variables,
        report_uninitialized_variables,
        assign,
        placeholder,
    )
    from tensorflow.keras.backend import set_session
except ImportError:
    from tensorflow.compat.v1 import (
        get_default_graph,
        report_uninitialized_variables,
        global_variables,
        initialize_variables,
        assign,
        placeholder,
    )
    from tensorflow.compat.v1.keras.backend import set_session

    tf.compat.v1.disable_eager_execution()

from federatedml.framework.weights import OrderDictWeights, Weights
from federatedml.nn.backend.tf_keras import losses
from federatedml.nn.homo_nn.nn_model import DataConverter, NNModel


def _zip_dir_as_bytes(path):
    with io.BytesIO() as io_bytes:
        with zipfile.ZipFile(io_bytes, "w", zipfile.ZIP_DEFLATED) as zipper:
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


def _init_session():
    sess = backend.get_session()
    get_default_graph()
    set_session(sess)
    return sess


def _modify_model_input_shape(nn_struct, input_shape):
    import copy
    import json

    if not input_shape:
        return json.dumps(nn_struct)

    if isinstance(input_shape, int):
        input_shape = [input_shape]
    else:
        input_shape = list(input_shape)

    struct = copy.deepcopy(nn_struct)
    if (
        not struct.get("config")
        or not struct["config"].get("layers")
        or not struct["config"]["layers"][0].get("config")
    ):
        return json.dumps(struct)

    if struct["config"]["layers"][0].get("config"):
        struct["config"]["layers"][0]["config"]["batch_input_shape"] = [
            None
        ] + input_shape
        return json.dumps(struct)
    else:
        return json.dump(struct)


def build_keras(nn_define, loss, optimizer, metrics, **kwargs):
    _init_session()
    nn_define_json = _modify_model_input_shape(
        nn_define, kwargs.get("input_shape", None)
    )
    model = tf.keras.models.model_from_json(nn_define_json, custom_objects={})
    keras_model = KerasNNModel(model)
    keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return keras_model


class KerasNNModel(NNModel):
    def __init__(self, model):
        self._sess: tf.Session = _init_session()
        self._model: tf.keras.Sequential = model
        self._trainable_weights = {
            v.name: v for v in self._model.trainable_weights
        }

        self._initialize_variables()
        self._loss = None

    def compile(self, loss, optimizer, metrics):
        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(
            **optimizer.kwargs
        )
        loss_fn = getattr(losses, loss)
        self._model.compile(optimizer=optimizer_instance, loss=loss_fn, metrics=metrics)

    def _initialize_variables(self):
        uninitialized_var_names = [
            bytes.decode(var)
            for var in self._sess.run(report_uninitialized_variables())
        ]
        uninitialized_vars = [
            var
            for var in global_variables()
            if var.name.split(":")[0] in uninitialized_var_names
        ]
        self._sess.run(initialize_variables(uninitialized_vars))

    @staticmethod
    def _trim_device_str(name):
        return name.split("/")[0]

    def get_model_weights(self) -> OrderDictWeights:
        return OrderDictWeights(self._sess.run(self._trainable_weights))

    def set_model_weights(self, weights: Weights):
        unboxed = weights.unboxed
        self._sess.run(
            [assign(v, unboxed[name]) for name, v in self._trainable_weights.items()]
        )

    def get_layer_by_index(self, layer_idx):
        return self._model.layers[layer_idx]

    def set_layer_weights_by_index(self, layer_idx, weights):
        self._model.layers[layer_idx].set_weights(weights)

    def get_input_gradients(self, X, y):
        return self._get_gradients(X, y, self._model.input)

    def get_trainable_gradients(self, X, y):
        return self._get_gradients(X, y, self._trainable_weights)

    def derivative_of_output_wrt_weights(self, X):
        gradient = gradients(self._model.output, self._model.trainable_variables)
        return self._sess.run(gradient, feed_dict={self._model.input: X})

    def apply_gradients(self, grads):
        update_ops = self._model.optimizer.apply_gradients(
            zip(grads, self._model.trainable_variables)
        )
        self._initialize_variables()
        self._sess.run(update_ops)

    def get_weight_gradients(self, X, y):
        return self._get_gradients(X, y, self._model.trainable_variables)

    def get_trainable_weights(self):
        return self._sess.run(self._model.trainable_variables)

    def get_loss(self):
        return self._loss

    def get_forward_loss_from_input(self, X, y):
        from federatedml.nn.hetero_nn.backend.tf_keras import losses

        y_true = placeholder(
            shape=self._model.output.shape, dtype=self._model.output.dtype
        )

        loss_fn = getattr(losses, self._model.loss_functions[0].fn.__name__)(
            y_true, self._model.output
        )
        return self._sess.run(loss_fn, feed_dict={self._model.input: X, y_true: y})

    def _get_gradients(self, X, y, variable):
        from federatedml.nn.hetero_nn.backend.tf_keras import losses

        y_true = placeholder(
            shape=self._model.output.shape, dtype=self._model.output.dtype
        )

        loss_fn = getattr(losses, self._model.loss_functions[0].fn.__name__)(
            y_true, self._model.output
        )
        gradient = gradients(loss_fn, variable)
        return self._sess.run(gradient, feed_dict={self._model.input: X, y_true: y})

    def set_learning_rate(self, learning_rate):
        assign_op = assign(self._model.optimizer.learning_rate, learning_rate)
        self._sess.run(assign_op)

    def train(self, data: tf.keras.utils.Sequence, **kwargs):
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        left_kwargs["callbacks"] = [History()]
        self._model.fit(x=data, epochs=epochs, verbose=1, shuffle=True, **left_kwargs)
        self._loss = left_kwargs["callbacks"][0].history["loss"]
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
        model_base = "./saved_model"
        if not os.path.exists(model_base):
            os.mkdir(model_base)
        model_path = f"{model_base}/{uuid.uuid1()}"
        os.mkdir(model_path)
        try:
            from tensorflow.keras.experimental import (
                export_saved_model as save_model,
            )

            save_model(self._model, model_path)

        except ImportError:
            from tensorflow.compat.v1 import saved_model

            saved_model.save(self._model, model_path)
            # from tensorflow.keras.models import save_model

        model_bytes = _zip_dir_as_bytes(model_path)

        return model_bytes

    @staticmethod
    def restore_model(
        model_bytes,
    ):  # todo: restore optimizer to support incremental learning
        model_base = "./restore_model"
        if not os.path.exists(model_base):
            os.mkdir(model_base)
        model_path = f"{model_base}/{uuid.uuid1()}"
        os.mkdir(model_path)
        with io.BytesIO(model_bytes) as bytes_io:
            with zipfile.ZipFile(bytes_io, "r", zipfile.ZIP_DEFLATED) as f:
                f.extractall(model_path)

        try:
            from tensorflow.keras.models import load_model

            # add custom objects
            from federatedml.nn.hetero_nn.backend.tf_keras.losses import (
                keep_predict_loss,
            )

            tf.keras.utils.get_custom_objects().update(
                {"keep_predict_loss": keep_predict_loss}
            )

        except ImportError:
            from tensorflow.keras.experimental import (
                load_from_saved_model as load_model,
            )

        model = load_model(f"{model_path}")

        return KerasNNModel(model)

    def export_optimizer_config(self):
        return self._model.optimizer.get_config()


class KerasSequenceData(tf.keras.utils.Sequence):
    def get_shape(self):
        return self.x_shape, self.y_shape

    def __init__(self, data_instances, batch_size, encode_label, label_mapping):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")

        _, one_data = data_instances.first()
        self.x_shape = one_data.features.shape
        num_label = len(label_mapping)
        print(label_mapping)

        if encode_label:
            if num_label > 2:
                self.y_shape = (num_label,)
                self.x = np.zeros((self.size, *self.x_shape))
                self.y = np.zeros((self.size, *self.y_shape))
                index = 0
                self._keys = []
                for k, inst in data_instances.collect():
                    self._keys.append(k)
                    self.x[index] = inst.features
                    self.y[index][label_mapping[inst.label]] = 1
                    index += 1
            else:
                raise ValueError(f"num_label is {num_label}")
        else:
            if num_label >= 2:
                self.y_shape = (1,)
            else:
                raise ValueError(f"num_label is {num_label}")
            self.x = np.zeros((self.size, *self.x_shape))
            self.y = np.zeros((self.size, *self.y_shape))
            index = 0
            self._keys = []
            for k, inst in data_instances.collect():
                self._keys.append(k)
                self.x[index] = inst.features
                self.y[index] = label_mapping[inst.label]
                index += 1

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
        return self.x[start:end], self.y[start:end]

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
