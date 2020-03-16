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
import os
import io
import copy
import typing
import zipfile
import tempfile

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Embedding, Lambda, Subtract, Concatenate, Multiply, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MSE as MSE

from arch.api.utils import log_utils
from federatedrec.utils import zip_dir_as_bytes
from federatedml.framework.weights import OrderDictWeights, Weights
from federatedrec.collaborative_memory.hetero_cmn.attention import VariableLengthMemoryLayer

LOGGER = log_utils.getLogger()


class CMNModel:
    """
    General Matrix Factorization model
    """

    def __init__(self, user_num=None, item_num=None, embedding_dim=32, hops=2, neg_count=4
                 , max_len=None, l2_coef=None, loss=None, optimizer=None, metrics=None, *args, **kwargs):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self._trainable_weights = None
        self._aggregate_weights = None
        self._predict_model = None
        self._model = None
        self._sess = None
        self.hops = hops
        self.neg_count = neg_count
        self.l2_coef = l2_coef
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.max_len = max_len

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
        self._model.fit(x=data, epochs=epochs, verbose=1, shuffle=True, **left_kwargs)
        return epochs * len(data)

    def _set_model(self, _model):
        """
        Set _model as trainning model.
        :param _model: training model.
        :return:
        """
        self._model = _model

    def _set_predict_model(self, _predict_model):
        """
        Set _predict_model as prediction model.
        :param _predict_model: prediction model.
        :return:
        """
        self._predict_model = _predict_model

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        """
        Apply func on model weights.
        :param func: operator to apply.
        :return: updated weights.
        """
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights

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
        self.session.run([tf.assign(v, unboxed[name]) for name, v in self._aggregate_weights.items()])

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

    def predict(self, data: tf.keras.utils.Sequence, **kwargs):
        """
        Predict on input data and return prediction results which used in prediction.
        :param data: input data.
        :return: prediction results.
        """
        return self._predict_model.predict(data)

    @classmethod
    def restore_model(cls, model_bytes, user_num, item_num, embedding_dim,
                      hops, max_len, l2_coef, loss, optimizer, metrics):
        """
        Restore model from model bytes.
        :param model_bytes: model bytes of saved model.
        :param user_num: user num
        :param item_num: item num
        :param embedding_dim: embedding dimension
        :return:restored model object.
        """
        LOGGER.info("begin restore_model")
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as file:
                    file.extractall(tmp_path)

            keras_model = tf.keras.experimental.load_from_saved_model(
                saved_model_path=tmp_path, custom_objects={"VariableLengthMemoryLayer": VariableLengthMemoryLayer})

        model = cls(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim
                    , hops=hops, max_len=max_len, l2_coef=l2_coef
                    , loss=loss, optimizer=optimizer, metrics=metrics)
        model._set_predict_model(keras_model)
        return model

    def export_model(self):
        """
        Export model to bytes.
        :return: bytes of saved model.
        """
        model_bytes = None
        with tempfile.TemporaryDirectory() as tmp_path:
            LOGGER.info(f"tmp_path: {tmp_path}")
            tf.keras.experimental.export_saved_model(
                self._predict_model, saved_model_path=tmp_path)

            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes

    def build(self, user_num, item_num, embedding_dim,
              max_len, hops, l2_coef, optimizer='rmsprop', loss='mse', metrics='mse'):

        """
        build network graph of model
        :param user_num: user num
        :param item_num: item num
        :param embedding_dim: embedding dimension
        :param optimizer: optimizer method
        :param loss:  loss methods
        :param metrics: metric methods
        :return:
        """
        sess = self.session
        users_input = Input(shape=(1,), dtype='int32', name='user_input')
        items_input = Input(shape=(1,), dtype='int32', name='item_input')
        neg_items_input = Input(shape=(1,), dtype='int32', name='neg_items_input')
        pos_length_input = Input(shape=(1,), dtype='int32', name='pos_length_input')
        pos_neighbors_input = Input(shape=(max_len,), dtype='int32', name='pos_neighbors_input')
        neg_length_input = Input(shape=(1,), dtype='int32', name='neg_length_input')
        neg_neighbors_input = Input(shape=(max_len,), dtype='int32', name='neg_neighbors_input')

        users = Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(users_input)
        items = Lambda(
            lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(tf.keras.backend.squeeze(x, -1)), item_num))(items_input)
        neg_items = Lambda(
            lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(tf.keras.backend.squeeze(x, -1)), item_num))(neg_items_input)
        pos_length = Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(pos_length_input)
        neg_length = Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(neg_length_input)

        user_embed_layer = Embedding(user_num, embedding_dim,
                                     embeddings_initializer=RandomNormal(stddev=0.1),
                                     name='MemoryEmbed')

        item_embed_layer = Embedding(item_num, embedding_dim,
                                     embeddings_initializer=RandomNormal(stddev=0.1),
                                     name='ItemMemory')

        user_external_embed_layer = Embedding(user_num, embedding_dim,
                                              embeddings_initializer=RandomNormal(stddev=0.1),
                                              name="MemoryOutput")
        # _mem_layer = VariableLengthMemoryLayer(name='UserMemoryLayer')
        _mem_layer = VariableLengthMemoryLayer(name='UserMemoryLayer', maxlen=max_len,
                                               hops=hops,
                                               embed_size=embedding_dim)
        output_module = tf.keras.Sequential([
            Dense(units=embedding_dim
                  , use_bias=False
                  , activation=tf.nn.relu
                  , kernel_initializer=RandomNormal(stddev=0.1)
                  , kernel_regularizer=l2(l2_coef)
                  , name='Layer'),
            Dense(units=1
                  , use_bias=False
                  , activation=tf.nn.relu
                  , kernel_initializer=RandomNormal(stddev=0.1)
                  , kernel_regularizer=l2(l2_coef)
                  , name='OutputVector')
        ])
        # positive
        cur_user_embed = user_embed_layer(users)
        cur_item_embed = item_embed_layer(items)
        neighbor_embed = user_embed_layer(pos_neighbors_input)
        neighbor_output_embed = user_external_embed_layer(pos_neighbors_input)

        query = (cur_user_embed, cur_item_embed)
        pos_inputs = [query, neighbor_embed, neighbor_output_embed, pos_length]
        neighbor = _mem_layer(inputs=pos_inputs).output

        pos_dot = Multiply()([cur_user_embed, cur_item_embed])
        pos_output = output_module(Concatenate(axis=1)([pos_dot, neighbor]))

        # negative
        neg_item_embed = item_embed_layer(neg_items)
        neg_neighbor_embed = user_embed_layer(neg_neighbors_input)
        neg_neighbor_output_embed = user_external_embed_layer(neg_neighbors_input)
        neg_query = (cur_user_embed, neg_item_embed)

        neg_inputs = [neg_query, neg_neighbor_embed, neg_neighbor_output_embed, neg_length]
        neg_neighbor = _mem_layer(inputs=neg_inputs).output
        neg_dot = Multiply()([cur_user_embed, neg_item_embed])
        neg_output = output_module(Concatenate(axis=1)([neg_dot, neg_neighbor]))

        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)
        loss_inst = Subtract(name="loss_layer")(inputs=[pos_output, neg_output])

        def cmn_loss(y_true, y_pred):
            eps = 1e-10
            loss = -tf.log(tf.nn.sigmoid(y_pred) + eps)
            return tf.reduce_mean(loss)

        self._model = Model(
            inputs=[users_input, items_input, neg_items_input, neg_length_input, neg_neighbors_input, pos_length_input,
                    pos_neighbors_input],
            outputs=[pos_output, neg_output, loss_inst])

        # model for prediction
        self._predict_model = Model(inputs=[users_input, items_input, pos_length_input, pos_neighbors_input],
                                    outputs=pos_output)
        LOGGER.info(f"model output names {self._model.output_names}")

        # self._model.compile(optimizer=optimizer_instance, loss=cmn_loss, metrics=metrics)
        self._model.compile(optimizer=optimizer_instance,
                            loss=[MSE, MSE, cmn_loss], metrics=["MSE", "MSE", "MSE"], loss_weights=[0.4, 0.4, 0.2])

        init = tf.initialize_all_variables()
        sess.run(init)

        # pick user_embedding for aggregating
        LOGGER.info(f"_trainable_weights: {self._model.trainable_weights}")
        self._trainable_weights = {v.name.split("/")[0]: v for v in self._model.trainable_weights}
        LOGGER.info(f"_trainable_weights: {self._trainable_weights}")
        # LOGGER.info(f"MemoryEmbed: {self.session.run(self._trainable_weights['MemoryEmbed'])}")

        self._aggregate_weights = {"MemoryOutput": self._trainable_weights["MemoryOutput"],
                                   "MemoryEmbed": self._trainable_weights["MemoryEmbed"]}

    @classmethod
    def build_model(cls, user_num, item_num, embedding_dim, hops, max_len
                    , l2_coef=0.01, loss="mse", optimizer="rmsprop"
                    , metrics="mse"):
        """
        build model
        :param l2_coef:
        :param user_num: user num
        :param item_num:  item num
        :param embedding_dim: embedding dimension
        :param loss: loss func
        :param optimizer: optimization methods
        :param metrics: metrics
        :return: model object
        """
        model = cls(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim
                    , hops=hops, max_len=max_len, l2_coef=l2_coef
                    , loss=loss, optimizer=optimizer, metrics=metrics)
        model.build(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim
                    , hops=hops, max_len=max_len, l2_coef=l2_coef
                    , loss=loss, optimizer=optimizer, metrics=metrics)
        return model

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

    def set_user_num(self, user_num):
        """
        set user num
        :param user_num:
        :return:
        """
        self.user_num = user_num

    def set_item_num(self, item_num):
        """
        set item num
        :param item_num:
        :return:
        """
        self.item_num = item_num
