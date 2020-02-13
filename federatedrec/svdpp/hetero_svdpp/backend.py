import io
import copy
import tempfile
import zipfile
import typing

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.initializers import RandomUniform, RandomNormal, Zeros
from tensorflow.python.keras.layers import Layer, Input, Embedding, Dot, Flatten, Dense, Dropout, Lambda, Add
from federatedml.framework.weights import OrderDictWeights, Weights
from federatedrec.utils import zip_dir_as_bytes
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class KerasSequenceData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, user_ids, item_ids, rating_matrix, batch_size):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")

        user_ids_map = {uid: i for i, uid in enumerate(user_ids)}
        item_ids_map = {iid: i for i, iid in enumerate(item_ids)}

        user_ids_num = len(user_ids)
        item_ids_num = len(item_ids)

        need_build_rm = True
        if rating_matrix is not None:
            self.rating_matrix = rating_matrix
            need_build_rm = False
        else:
            self.rating_matrix = np.zeros((user_ids_num, item_ids_num))

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
            if need_build_rm:
                self.rating_matrix[user_ids_map[uid]][item_ids_map[iid]] = item_ids_map[iid] + 1
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

        user_ids_indexes = self.x[start: end, 0]
        item_ids_indexes = self.x[start: end, 1]
        implicit_feedback = self.rating_matrix[user_ids_indexes.astype(np.int32).flatten(), :]

        return [user_ids_indexes, item_ids_indexes, implicit_feedback], self.y[start: end]

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys


class KerasSeqDataConverter:
    """
    Keras Sequence Data Converter
    """

    @staticmethod
    def convert(data, user_ids, item_ids, batch_size, rating_matrix=None):
        return KerasSequenceData(data, user_ids, item_ids, rating_matrix, batch_size)


class ConstantLayer(Layer):
    def __init__(self, mu, **kwargs):
        self.mu = mu
        self.output_dim = 1
        super(ConstantLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a weight variable for this layer.
        self.kernel = self.add_weight(name='mu',
                                      shape=(self.output_dim,),
                                      initializer=tf.constant_initializer(self.mu),
                                      trainable=False)
        super(ConstantLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x + self.kernel

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super(ConstantLayer, self).get_config()
        config['mu'] = self.mu
        return config


class MeanPoolingLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanPoolingLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        """Mask is not passed to next layer."""
        return None

    def call(self, x, mask=None):
        """1, mask is a bool type tensor, need casting before compute.
           2, mask shape in 2 dimension (batch_size, feature_dimension)
        """
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            mask = tf.cast(mask, tf.float32)
            x = x * mask
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0], input_shape[2]
        return output_shape


class SVDppModel:
    def __init__(self, user_ids=None, item_ids=None, rating_matrix=None, embedding_dim=None, mu=None):
        """
        :param user_ids: user ids
        :param item_ids: item ids
        :param embedding_dim: embedding dimension
        :param mu: average rate of training data
        """
        if user_ids is not None:
            self.user_num = len(user_ids)
        if item_ids is not None:
            self.item_num = len(item_ids)
        self.rating_matrix = rating_matrix
        self.embedding_dim = embedding_dim
        self._sess = None
        self._trainable_weights = None
        self._aggregate_weights = None
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.mu = mu

    def _build(self, lambda_u=0.0001, lambda_v=0.0001, optimizer='rmsprop',
               loss='mse', metrics='mse', initializer='uniform'):
        # init session on first time ref
        sess = self.session

        # user embedding
        user_InputLayer = Input(shape=(1,), dtype='int32', name='user_input')
        user_EmbeddingLayer = Embedding(input_dim=self.user_num,
                                        output_dim=self.embedding_dim,
                                        input_length=1,
                                        name='user_embedding',
                                        embeddings_regularizer=l2(lambda_u),
                                        embeddings_initializer=initializer)(user_InputLayer)
        user_EmbeddingLayer = Flatten(name='user_flatten')(user_EmbeddingLayer)

        # implicit feedback
        feedback_InputLayer = Input(shape=(None,), dtype='int32', name='implicit_feedback')
        feedback_EmbeddingLayer = Embedding(input_dim=self.item_num + 1,
                                            output_dim=self.embedding_dim,
                                            name='implicit_feedback_embedding',
                                            embeddings_regularizer=l2(lambda_v),
                                            embeddings_initializer=initializer,
                                            mask_zero=True)(feedback_InputLayer)
        feedback_EmbeddingLayer = MeanPoolingLayer()(feedback_EmbeddingLayer)

        user_EmbeddingLayer = Add()([user_EmbeddingLayer, feedback_EmbeddingLayer])

        # user bias
        user_BiasLayer = Embedding(input_dim=self.user_num, output_dim=1, input_length=1,
                                   name='user_bias', embeddings_regularizer=l2(lambda_u),
                                   embeddings_initializer=Zeros())(user_InputLayer)
        user_BiasLayer = Flatten()(user_BiasLayer)

        # item embedding
        item_InputLayer = Input(shape=(1,), dtype='int32', name='item_input')
        item_EmbeddingLayer = Embedding(input_dim=self.item_num, output_dim=self.embedding_dim, input_length=1,
                                        name='item_embedding', embeddings_regularizer=l2(lambda_v),
                                        embeddings_initializer=RandomNormal(mean=0, stddev=1))(item_InputLayer)
        item_EmbeddingLayer = Flatten(name='item_flatten')(item_EmbeddingLayer)

        # item bias
        item_BiasLayer = Embedding(input_dim=self.item_num, output_dim=1, input_length=1,
                                   name='item_bias', embeddings_regularizer=l2(lambda_v),
                                   embeddings_initializer=Zeros())(item_InputLayer)
        item_BiasLayer = Flatten()(item_BiasLayer)

        # rating prediction
        dotLayer = Dot(axes=-1, name='dot_layer')([user_EmbeddingLayer, item_EmbeddingLayer])

        # add mu, user bias and item bias
        dotLayer = ConstantLayer(mu=self.mu)(dotLayer)
        dotLayer = Add()([dotLayer, user_BiasLayer])
        dotLayer = Add()([dotLayer, item_BiasLayer])

        # create model
        self._model = Model(inputs=[user_InputLayer, item_InputLayer, feedback_InputLayer], outputs=[dotLayer])

        # compile model
        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)
        losses = getattr(tf.keras.losses, loss)
        self._model.compile(optimizer=optimizer_instance,
                            loss=losses, metrics=metrics)
        # pick user_embedding and user_bias for aggregating
        self._trainable_weights = {v.name.split("/")[0]: v for v in self._model.trainable_weights}
        LOGGER.debug(f"trainable weights {self._trainable_weights}")
        self._aggregate_weights = {"user_embedding": self._trainable_weights["user_embedding"],
                                   "user_bias": self._trainable_weights["user_bias"]}

    def train(self, data: tf.keras.utils.Sequence, **kwargs):
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        self._model.fit(x=data, epochs=epochs, verbose=1, shuffle=True, **left_kwargs)
        return epochs * len(data)

    def _set_model(self, _model):
        self._model = _model

    @classmethod
    def build_model(cls, user_ids, item_ids, rating_matrix, embedding_dim, loss, optimizer, metrics, mu):
        model = cls(user_ids, item_ids, rating_matrix, embedding_dim, mu)
        model._build(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    @classmethod
    def restore_model(cls, model_bytes):  # todo: restore optimizer to support incremental learning
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as f:
                    f.extractall(tmp_path)

            # try:
            #     keras_model = tf.keras.models.load_model(filepath=tmp_path,
            #                                              custom_objects={'ConstantLayer': ConstantLayer,
            #                                                              'MeanPoolingLayer': MeanPoolingLayer})
            # except IOError:
            #     import warnings
            #     warnings.warn('loading the model as SavedModel is still in experimental stages. '
            #                   'trying tf.keras.experimental.load_from_saved_model...')
            keras_model = \
                tf.keras.experimental.load_from_saved_model(saved_model_path=tmp_path,
                                                            custom_objects={'ConstantLayer': ConstantLayer,
                                                                            'MeanPoolingLayer': MeanPoolingLayer})
        model = cls()
        model._set_model(keras_model)
        return model

    def export_model(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            # try:
            #     # LOGGER.info("Model saved with model.save method.")
            #     tf.keras.models.save_model(self._model, filepath=tmp_path, save_format="tf")
            # except NotImplementedError:
            #     import warnings
            #     warnings.warn('Saving the model as SavedModel is still in experimental stages. '
            #                   'trying tf.keras.experimental.export_saved_model...')
            tf.keras.experimental.export_saved_model(self._model, saved_model_path=tmp_path)

            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights

    @property
    def session(self):
        if self._sess is None:
            sess = tf.Session()
            # tf.get_default_graph()
            set_session(sess)
            self._sess = sess
        return self._sess

    def get_model_weights(self) -> OrderDictWeights:
        return OrderDictWeights(self.session.run(self._aggregate_weights))

    def set_model_weights(self, weights: Weights):
        unboxed = weights.unboxed
        self.session.run([tf.assign(v, unboxed[name]) for name, v in self._aggregate_weights.items()])

    def evaluate(self, data: tf.keras.utils.Sequence):
        names = self._model.metrics_names
        values = self._model.evaluate(x=data, verbose=1)
        if not isinstance(values, list):
            values = [values]
        return dict(zip(names, values))

    def predict(self, data: tf.keras.utils.Sequence, **kwargs):
        return self._model.predict(data)

    def set_user_ids(self, user_ids):
        self.user_ids = user_ids

    def set_item_ids(self, item_ids):
        self.item_ids = item_ids

    def set_rating_matrix(self, rating_matrix):
        self.rating_matrix = rating_matrix
