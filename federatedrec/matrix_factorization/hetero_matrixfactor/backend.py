#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import copy
import io
import tempfile
import typing
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Dot, Embedding, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from federatedml.framework.weights import OrderDictWeights, Weights
from federatedrec.utils import zip_dir_as_bytes


class KerasSequenceData(tf.keras.utils.Sequence):
    """
    Keras Sequence Data Class.
    """
    def __init__(self, data_instances, user_ids, item_ids, batch_size):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")

        user_ids_map = {uid: i for i, uid in enumerate(user_ids)}
        item_ids_map = {iid: i for i, iid in enumerate(item_ids)}

        self.x = np.zeros((self.size, 2))
        self.y = np.zeros((self.size, 1))
        self._keys = []
        for index, (k, inst) in enumerate(data_instances.collect()):
            self._keys.append(k)
            uid = inst.features.get_data(0)
            iid = inst.features.get_data(1)
            rate = float(inst.features.get_data(2))
            self.x[index] = [user_ids_map[uid], item_ids_map[iid]]
            self.y[index] = rate

        self.batch_size = batch_size if batch_size > 0 else self.size

    def __getitem__(self, index):
        """
        Gets batch at position `index`.
        :param index: position of the batch in the Sequence.
        :return: A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return [self.x[start: end, 0],
                self.x[start: end, 1]], self.y[start: end]

    def __len__(self):
        """Number of batch in the Sequence.
        "return: The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        """
        Return keys of data.
        :return: keys of data.
        """
        return self._keys


class KerasSeqDataConverter:
    """
    Keras Sequence Data Converter.
    """
    @staticmethod
    def convert(data, user_ids, item_ids, batch_size):
        """
        Convert input to KerasSequenceData object.
        :param data: DTable of data
        :param user_ids: user ids.
        :param item_ids: item ids.
        :param batch_size: batch size.
        :return:
        """
        return KerasSequenceData(data, user_ids, item_ids, batch_size)


class MFModel:
    """
    Matrix Factorization Model Class.
    """
    def __init__(self, user_ids=None, item_ids=None, embedding_dim=None):
        if user_ids is not None:
            self.user_num = len(user_ids)
        if item_ids is not None:
            self.item_num = len(item_ids)
        self.embedding_dim = embedding_dim
        self._sess = None
        self._model = None
        self._trainable_weights = None
        self._aggregate_weights = None
        self.user_ids = user_ids
        self.item_ids = item_ids

    def build(self, lambda_u=0.0001, lambda_v=0.0001, optimizer='rmsprop',
              loss='mse', metrics='mse', initializer='uniform'):
        """
        Init session and create model architecture.
        :param lambda_u: lambda value of l2 norm for user embeddings.
        :param lambda_v: lambda value of l2 norm for item embeddings.
        :param optimizer: optimizer type.
        :param loss: loss type.
        :param metrics: evaluation metrics.
        :param initializer: initializer of embedding
        :return:
        """
        # init session on first time ref
        sess = self.session
        # user embedding
        user_input_layer = Input(shape=(1,), dtype='int32', name='user_input')
        user_embedding_layer = Embedding(
            input_dim=self.user_num,
            output_dim=self.embedding_dim,
            input_length=1,
            name='user_embedding',
            embeddings_regularizer=l2(lambda_u),
            embeddings_initializer=initializer)(user_input_layer)
        user_embedding_layer = Flatten(name='user_flatten')(user_embedding_layer)

        # item embedding
        item_input_layer = Input(shape=(1,), dtype='int32', name='item_input')
        item_embedding_layer = Embedding(
            input_dim=self.item_num,
            output_dim=self.embedding_dim,
            input_length=1,
            name='item_embedding',
            embeddings_regularizer=l2(lambda_v),
            embeddings_initializer=initializer)(item_input_layer)
        item_embedding_layer = Flatten(name='item_flatten')(item_embedding_layer)

        # rating prediction
        dot_layer = Dot(axes=-1,
                        name='dot_layer')([user_embedding_layer,
                                           item_embedding_layer])
        self._model = Model(
            inputs=[user_input_layer, item_input_layer], outputs=[dot_layer])

        # compile model
        optimizer_instance = getattr(
            tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)
        losses = getattr(tf.keras.losses, loss)
        self._model.compile(optimizer=optimizer_instance,
                            loss=losses, metrics=metrics)
        # pick user_embedding for aggregating
        self._trainable_weights = {v.name.split(
            "/")[0]: v for v in self._model.trainable_weights}
        self._aggregate_weights = {
            "user_embedding": self._trainable_weights["user_embedding"]}

    def train(self, data: tf.keras.utils.Sequence, **kwargs):
        """
        Train model on input data.
        :param data: input data.
        :param kwargs: other params.
        :return: Training steps.
        """
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        self._model.fit(x=data, epochs=epochs, verbose=1,
                        shuffle=True, **left_kwargs)
        return epochs * len(data)

    def set_model(self, _model):
        """
        Set _model as input model.
        :param _model: input model.
        :return:
        """
        self._model = _model

    @classmethod
    def build_model(cls, user_ids, item_ids, embedding_dim,
                    loss, optimizer, metrics):
        """
        Build and return model object.
        :param user_ids: user ids.
        :param item_ids: item ids.
        :param embedding_dim: embedding dimension.
        :param loss: loss type.
        :param optimizer: optimizer type.
        :param metrics: evaluation metrics.
        :return: model object.
        """
        model = cls(user_ids, item_ids, embedding_dim)
        model.build(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    @classmethod
    # todo: restore optimizer to support incremental learning
    def restore_model(cls, model_bytes):
        """
        Restore model from model bytes.
        :param model_bytes: model bytes of saved model.
        :return: restored model object.
        """
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as file:
                    file.extractall(tmp_path)

            keras_model = tf.keras.experimental.load_from_saved_model(
                    saved_model_path=tmp_path)
        model = cls()
        model.set_model(keras_model)
        return model

    def export_model(self):
        """
        Export model to bytes.
        :return: bytes of saved model.
        """
        with tempfile.TemporaryDirectory() as tmp_path:
            tf.keras.experimental.export_saved_model(
                    self._model, saved_model_path=tmp_path)

            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        """
        Apply func on model weights.
        :param func: operator to apply.
        :return: updated weights.
        """
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights

    @property
    def session(self):
        """
        If session not created, then init a tensorflow session and return.
        :return: tensorflow session.
        """
        if self._sess is None:
            sess = tf.Session()
            tf.get_default_graph()
            set_session(sess)
            self._sess = sess
        return self._sess

    def get_model_weights(self) -> OrderDictWeights:
        """
        Return model's weights as OrderDictWeights.
        :return: model's weights.
        """
        return OrderDictWeights(self.session.run(self._aggregate_weights))

    def set_model_weights(self, weights: Weights):
        """
        Set model's weights with input weights.
        :param weights: input weights.
        :return: updated weights.
        """
        unboxed = weights.unboxed
        self.session.run([tf.assign(v, unboxed[name])
                          for name, v in self._aggregate_weights.items()])

    def evaluate(self, data: tf.keras.utils.Sequence):
        """
        Evaluate on input data and return evaluation results.
        :param data: input data sequence.
        :return: evaluation results.
        """
        names = self._model.metrics_names
        values = self._model.evaluate(x=data, verbose=1)
        if not isinstance(values, list):
            values = [values]
        return dict(zip(names, values))

    def predict(self, data: tf.keras.utils.Sequence):
        """
        Predict on input data and return prediction results.
        :param data: input data.
        :return: prediction results.
        """
        return self._model.predict(data)

    def set_user_ids(self, user_ids):
        """
        set user ids.
        :param user_ids:
        :return:
        """
        self.user_ids = user_ids

    def set_item_ids(self, item_ids):
        """
        set item ids
        :param item_ids:
        :return:
        """
        self.item_ids = item_ids
