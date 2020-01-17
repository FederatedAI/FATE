import io
import os
import copy
import typing
import zipfile
import tempfile

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers import Input, Embedding, Multiply, Dense, Lambda, Subtract, Dot
from arch.api.utils import log_utils
from federatedrec.utils import zip_dir_as_bytes
from federatedml.framework.weights import OrderDictWeights, Weights
from tensorflow.keras.losses import MSE as MSE

LOGGER = log_utils.getLogger()


class KerasModel:
    def __init__(self):
        self._aggregate_weights = None
        self.session = None
        self._predict_model = None

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

    def _set_predict_model(self, _predict_model):
        self._predict_model = _predict_model

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights

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
        if self._predict_model is not None:
            return self._predict_model.predict(data)
        else:
            return self._model.predict(data)

    @classmethod
    def restore_model(cls, model_bytes):  # todo: restore optimizer to support incremental learning
        model = cls()
        LOGGER.info("begin restore_model")
        keras_model = None
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as f:
                    f.extractall(tmp_path)

            os.system(f"echo 'test tmp_path:' {tmp_path}; ls {tmp_path}; du -h {tmp_path}")
            try:
                keras_model = tf.keras.models.load_model(filepath=tmp_path)
            except IOError:
                import warnings
                warnings.warn('loading the model as SavedModel is still in experimental stages. '
                              'trying tf.keras.experimental.load_from_saved_model...')
                keras_model = tf.keras.experimental.load_from_saved_model(saved_model_path=tmp_path)
        model._set_predict_model(keras_model)
        return model

    def export_model(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            LOGGER.info(f"tmp_path: {tmp_path}")
            try:
                LOGGER.info(f"predict model is None: {self._predict_model == None}")
                tf.keras.models.save_model(self._predict_model, filepath=tmp_path, save_format="tf", include_optimizer=False)
                # tf.keras.experimental.export_saved_model(self._predict_model, saved_model_path=tmp_path)
            except Exception as e:
                import warnings
                warnings.warn('Saving the model as SavedModel is still in experimental stages. '
                              'trying tf.keras.experimental.export_saved_model...')

                tf.keras.experimental.export_saved_model(self._predict_model, saved_model_path=tmp_path)
            os.system(f"echo 'test tmp_path:' {tmp_path}; ls {tmp_path}; du -h {tmp_path}")
            LOGGER.info(f"export saved model at path: {tmp_path}")
            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes


class GMFModel(KerasModel):
    def __init__(self, user_ids=None, item_ids=None, embedding_dim=10, l2_coef=0.01, user_num=10000):
        super().__init__()
        if user_ids is not None:
            self.user_num = max(max(user_ids), max(len(user_ids), user_num)) + 100

        if item_ids is not None:
            self.item_num = len(item_ids)
        self.embedding_dim = embedding_dim
        self._sess = None
        self._trainable_weights = None
        self._aggregate_weights = None
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.session = self.init_session()
        self._embedding_initializer = TruncatedNormal(stddev=0.1)
        self._initializer = TruncatedNormal(stddev=0.1)
        self._regularizer = l2(l2_coef)

    def _build(self, optimizer='rmsprop', loss='mse', metrics='mse'):
        users_input = Input(shape=(1,), dtype='int32', name='user_input')
        items_input = Input(shape=(1,), dtype='int32', name='item_input')
        neg_items_input = Input(shape=(1,), dtype='int32', name='neg_items_input')

        users = Lambda(lambda x: tf.squeeze(x, 1))(users_input)
        items = Lambda(
            lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(tf.squeeze(x, 1)), self.item_num))(items_input)
        neg_items = Lambda(
            lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(tf.squeeze(x, 1)), self.item_num))(neg_items_input)
        user_embed_layer = Embedding(self.user_num, self.embedding_dim,
                                     embeddings_initializer=self._embedding_initializer,
                                     name='user_embedding')

        item_embed_layer = Embedding(self.item_num, self.embedding_dim,
                                     embeddings_initializer=self._embedding_initializer,
                                     name='item_embedding')

        cur_user_embed = user_embed_layer(users)
        cur_item_embed = item_embed_layer(items)
        neg_item_embed = item_embed_layer(neg_items)

        LOGGER.info(f"users shapes: {users_input.shape}")
        LOGGER.info(f"items shapes: {items_input.shape}")
        LOGGER.info(f"neg_items shapes: {neg_items_input.shape}")
        LOGGER.info(f"embedding shapes, user_embedding: {cur_user_embed.shape}")
        LOGGER.info(f"embedding shapes, item_embedding: {cur_item_embed.shape}")
        LOGGER.info(f"embedding shapes, neg_item_embedding: {neg_item_embed.shape}")

        # output_layer = Dense(units=1
        #                      , use_bias=False
        #                      , activation=tf.nn.relu
        #                      , kernel_initializer=self._initializer
        #                      , kernel_regularizer=self._regularizer
        #                      , name='OutputVector')
        #
        # pos_mul = Multiply(name='pos_mul_layer')(inputs=[cur_user_embed, cur_item_embed])
        # neg_mul = Multiply(name='neg_mul_layer')(inputs=[cur_user_embed, neg_item_embed])
        # pos_output = output_layer(pos_mul)
        # neg_output = output_layer(neg_mul)
        pos_output = Dot(axes=-1, name="pos_dot")([cur_user_embed, cur_item_embed])
        neg_output = Dot(axes=-1, name="neg_dot")([cur_user_embed, neg_item_embed])
        loss_inst = Subtract(name="loss_layer")(inputs=[pos_output, neg_output])

        def gmf_loss(y_true, y_pred):
            eps = 1e-10
            loss = -tf.log(tf.nn.sigmoid(y_pred) + eps)
            return tf.reduce_mean(loss)

        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)

        LOGGER.info(f"loss_inst type: {type(loss_inst)}, shape: {loss_inst.shape}")

        self._model = Model(inputs=[users_input, items_input, neg_items_input],
                            outputs=[pos_output, neg_output, loss_inst])
        # model for prediction
        LOGGER.info(f"model output names {self._model.output_names}")
        self._predict_model = Model(inputs=[users_input, items_input],
                                    outputs=pos_output)

        self._model.compile(optimizer=optimizer_instance,
                            loss=[MSE, MSE, gmf_loss], metrics=["MSE"], loss_weights=[0.3, 0.3, 0.4])

        # pick user_embedding for aggregating
        self._trainable_weights = {v.name.split("/")[0]: v for v in self._model.trainable_weights}
        self._aggregate_weights = {"user_embedding": self._trainable_weights["user_embedding"]}
        LOGGER.info(f"finish building model, in {self.__class__.__name__} _build function")

    @classmethod
    def build_model(cls, user_ids, item_ids, embedding_dim, loss, optimizer, metrics, user_num):
        model = cls(user_ids=user_ids, item_ids=item_ids, embedding_dim=embedding_dim, user_num=user_num)
        model._build(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def init_session(self):
        sess = tf.Session()
        tf.get_default_graph()
        set_session(sess)
        return sess

    def set_user_ids(self, user_ids):
        self.user_ids = user_ids

    def set_item_ids(self, item_ids):
        self.item_ids = item_ids
