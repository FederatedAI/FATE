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

import json

from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.nn.hetero_nn.hetero_nn_model import HeteroNNGuestModel
from federatedml.nn.hetero_nn.hetero_nn_model import HeteroNNHostModel
from federatedml.nn.hetero_nn.model.hetero_nn_bottom_model import HeteroNNBottomModel
from federatedml.nn.hetero_nn.model.hetero_nn_top_model import HeteroNNTopModel
from federatedml.nn.hetero_nn.model.interactive_layer import InterActiveGuestDenseLayer
from federatedml.nn.hetero_nn.model.interactive_layer import InteractiveHostDenseLayer
from federatedml.nn.homo_nn import nn_model
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNModelMeta
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import OptimizerParam
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNModelParam
from federatedml.util import LOGGER
from federatedml.nn.hetero_nn.strategy.selector import SelectorFactory


class HeteroNNKerasGuestModel(HeteroNNGuestModel):
    def __init__(self, hetero_nn_param):
        super(HeteroNNKerasGuestModel, self).__init__()

        self.bottom_model = None
        self.interactive_model = None
        self.top_model = None
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None
        self.config_type = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.hetero_nn_param = None
        self.transfer_variable = None
        self.model_builder = None
        self.bottom_model_input_shape = 0
        self.top_model_input_shape = None

        self.batch_size = None

        self.is_empty = False

        self.set_nn_meta(hetero_nn_param)
        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)
        self.data_converter = KerasSequenceDataConverter()

        self.selector = SelectorFactory.get_selector(hetero_nn_param.selector_param.method,
                                                     hetero_nn_param.selector_param.selective_size,
                                                     beta=hetero_nn_param.selector_param.beta,
                                                     random_rate=hetero_nn_param.selector_param.random_state,
                                                     min_prob=hetero_nn_param.selector_param.min_prob)

    def set_nn_meta(self, hetero_nn_param):
        self.bottom_nn_define = hetero_nn_param.bottom_nn_define
        self.top_nn_define = hetero_nn_param.top_nn_define
        self.interactive_layer_define = hetero_nn_param.interactive_layer_define
        self.config_type = hetero_nn_param.config_type
        self.optimizer = hetero_nn_param.optimizer
        self.loss = hetero_nn_param.loss
        self.hetero_nn_param = hetero_nn_param
        self.batch_size = hetero_nn_param.batch_size

    def set_empty(self):
        self.is_empty = True

    def train(self, x, y, epoch, batch_idx):
        if self.batch_size == -1:
            self.batch_size = x.shape[0]

        if not self.is_empty:
            if self.bottom_model is None:
                self.bottom_model_input_shape = x.shape[1]
                self._build_bottom_model()

            guest_bottom_output = self.bottom_model.forward(x)
        else:
            guest_bottom_output = None

        if self.interactive_model is None:
            self._build_interactive_model()

        interactive_output = self.interactive_model.forward(guest_bottom_output, epoch, batch_idx, train=True)

        if self.top_model is None:
            self.top_model_input_shape = int(interactive_output.shape[1])
            self._build_top_model()

        selective_ids, gradients, loss = self.top_model.train_and_get_backward_gradient(interactive_output, y)

        interactive_layer_backward = self.interactive_model.backward(gradients, selective_ids, epoch, batch_idx)

        if not self.is_empty:
            self.bottom_model.backward(x, interactive_layer_backward, selective_ids)

        return loss

    def predict(self, x):
        if not self.is_empty:
            guest_bottom_output = self.bottom_model.predict(x)
        else:
            guest_bottom_output = None

        interactive_output = self.interactive_model.forward(guest_bottom_output, train=False)
        preds = self.top_model.predict(interactive_output)

        return preds

    def evaluate(self, x, y, epoch, batch):
        if not self.is_empty:
            guest_bottom_output = self.bottom_model.predict(x)
        else:
            guest_bottom_output = None

        interactive_output = self.interactive_model.forward(guest_bottom_output, epoch, batch, train=False)
        metrics = self.top_model.evaluate(interactive_output, y)

        return metrics

    def get_hetero_nn_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.is_empty = self.is_empty
        if not self.is_empty:
            model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.top_saved_model_bytes = self.top_model.export_model()
        model_param.interactive_layer_param.CopyFrom(self.interactive_model.export_model())

        model_param.bottom_model_input_shape = self.bottom_model_input_shape
        model_param.top_model_input_shape = self.top_model_input_shape

        return model_param

    def set_hetero_nn_model_param(self, model_param):
        self.is_empty = model_param.is_empty
        self.top_model_input_shape = model_param.top_model_input_shape
        self.bottom_model_input_shape = model_param.bottom_model_input_shape
        if not self.is_empty:
            self._restore_bottom_model(model_param.bottom_saved_model_bytes)

        self._restore_interactive_model(model_param.interactive_layer_param)

        self._restore_top_model(model_param.top_saved_model_bytes)

    def get_hetero_nn_model_meta(self):
        model_meta = HeteroNNModelMeta()
        model_meta.config_type = self.config_type

        if self.config_type == "nn":
            for layer in self.bottom_nn_define:
                model_meta.bottom_nn_define.append(json.dumps(layer))

            for layer in self.top_nn_define:
                model_meta.top_nn_define.append(json.dumps(layer))
        elif self.config_type == "keras":
            model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))
            model_meta.top_nn_define.append(json.dumps(self.top_nn_define))

        model_meta.interactive_layer_define = json.dumps(self.interactive_layer_define)
        model_meta.interactive_layer_lr = self.hetero_nn_param.interactive_layer_lr

        model_meta.loss = self.loss

        """
        for metric in self.metrics:
            model_meta.metrics.append(metric)
        """

        optimizer_param = OptimizerParam()
        optimizer_param.optimizer = self.optimizer.optimizer
        optimizer_param.kwargs = json.dumps(self.optimizer.kwargs)

        model_meta.optimizer_param.CopyFrom(optimizer_param)

        return model_meta

    def set_hetero_nn_model_meta(self, model_meta):
        self.config_type = model_meta.config_type

        if self.config_type == "nn":
            self.bottom_nn_define = []
            self.top_nn_define = []

            for layer in model_meta.bottom_nn_define:
                self.bottom_nn_define.append(json.loads(layer))

            for layer in model_meta.top_nn_define:
                self.top_nn_define.append(json.loads(layer))
        elif self.config_type == 'keras':
            self.bottom_nn_define = json.loads(model_meta.bottom_nn_define[0])
            self.top_nn_define = json.loads(model_meta.top_nn_define[0])

        self.interactive_layer_define = json.loads(model_meta.interactive_layer_define)
        self.loss = model_meta.loss

        self.metrics = []
        for metric in self.metrics:
            self.metrics.append(metric)

        if self.optimizer is None:
            from types import SimpleNamespace
            self.optimizer = SimpleNamespace(optimizer=None, kwargs={})
            self.optimizer.optimizer = model_meta.optimizer_param.optimizer
            self.optimizer.kwargs = json.loads(model_meta.optimizer_param.kwargs)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition=1):
        self.partition = partition
        if self.interactive_model is not None:
            self.interactive_model.set_partition(self.partition)

    def _build_bottom_model(self):
        self.bottom_model = HeteroNNBottomModel(input_shape=self.bottom_model_input_shape,
                                                optimizer=self.optimizer,
                                                layer_config=self.bottom_nn_define,
                                                model_builder=self.model_builder)

        self.bottom_model.set_data_converter(self.data_converter)
        if self.selector:
            self.bottom_model.set_backward_select_strategy()
            self.bottom_model.set_batch(self.batch_size)

    def _restore_bottom_model(self, model_bytes):
        self._build_bottom_model()
        self.bottom_model.restore_model(model_bytes)

    def _build_top_model(self):
        self.top_model = HeteroNNTopModel(input_shape=self.top_model_input_shape,
                                          optimizer=self.optimizer,
                                          layer_config=self.top_nn_define,
                                          loss=self.loss,
                                          metrics=self.metrics,
                                          model_builder=self.model_builder)

        self.top_model.set_data_converter(self.data_converter)

        if self.selector:
            self.top_model.set_backward_selector_strategy(selector=self.selector)
            self.top_model.set_batch(self.batch_size)

    def _restore_top_model(self, model_bytes):
        self._build_top_model()
        self.top_model.restore_model(model_bytes)

    def _build_interactive_model(self):
        self.interactive_model = InterActiveGuestDenseLayer(self.hetero_nn_param,
                                                            self.interactive_layer_define,
                                                            model_builder=self.model_builder)

        self.interactive_model.set_transfer_variable(self.transfer_variable)
        self.interactive_model.set_partition(self.partition)
        if self.selector:
            self.interactive_model.set_backward_select_strategy()

    def _restore_interactive_model(self, interactive_model_param):
        self._build_interactive_model()

        self.interactive_model.restore_model(interactive_model_param)

    def warm_start(self):
        self.bottom_model.recompile(self.optimizer)
        self.top_model.recompile(self.loss, self.optimizer, self.metrics)


class HeteroNNKerasHostModel(HeteroNNHostModel):
    def __init__(self, hetero_nn_param):
        super(HeteroNNKerasHostModel, self).__init__()

        self.bottom_model_input_shape = None
        self.bottom_model = None
        self.interactive_model = None

        self.bottom_nn_define = None
        self.config_type = None
        self.optimizer = None
        self.hetero_nn_param = None

        self.batch_size = None
        self.set_nn_meta(hetero_nn_param)

        self.model_builder = nn_model.get_nn_builder(config_type=self.config_type)
        self.data_converter = KerasSequenceDataConverter()

        self.transfer_variable = None

        self.selector = SelectorFactory.get_selector(hetero_nn_param.selector_param.method,
                                                     hetero_nn_param.selector_param.selective_size,
                                                     beta=hetero_nn_param.selector_param.beta,
                                                     random_rate=hetero_nn_param.selector_param.random_state,
                                                     min_prob=hetero_nn_param.selector_param.min_prob)

    def set_nn_meta(self, hetero_nn_param):
        self.bottom_nn_define = hetero_nn_param.bottom_nn_define
        self.config_type = hetero_nn_param.config_type
        self.optimizer = hetero_nn_param.optimizer
        self.hetero_nn_param = hetero_nn_param
        self.batch_size = hetero_nn_param.batch_size

    def _build_bottom_model(self):
        self.bottom_model = HeteroNNBottomModel(input_shape=self.bottom_model_input_shape,
                                                optimizer=self.optimizer,
                                                layer_config=self.bottom_nn_define,
                                                model_builder=self.model_builder)

        self.bottom_model.set_data_converter(self.data_converter)

    def _restore_bottom_model(self, model_bytes):
        self._build_bottom_model()
        self.bottom_model.restore_model(model_bytes)

    def _build_interactive_model(self):
        self.interactive_model = InteractiveHostDenseLayer(self.hetero_nn_param)

        self.interactive_model.set_transfer_variable(self.transfer_variable)
        self.interactive_model.set_partition(self.partition)

    def _restore_interactive_model(self, interactive_layer_param):
        self._build_interactive_model()
        self.interactive_model.restore_model(interactive_layer_param)

    def warm_start(self):
        self.bottom_model.recompile(self.optimizer)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition=1):
        self.partition = partition
        if self.interactive_model is not None:
            self.interactive_model.set_partition(self.partition)

        LOGGER.debug("set_partition, partition num is {}".format(self.partition))

    def get_hetero_nn_model_meta(self):
        model_meta = HeteroNNModelMeta()

        model_meta.config_type = self.config_type

        if self.config_type == "nn":
            for layer in self.bottom_nn_define:
                model_meta.bottom_nn_define.append(json.dumps(layer))

        elif self.config_type == "keras":
            model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))

        model_meta.interactive_layer_lr = self.hetero_nn_param.interactive_layer_lr

        optimizer_param = OptimizerParam()
        optimizer_param.optimizer = self.optimizer.optimizer
        optimizer_param.kwargs = json.dumps(self.optimizer.kwargs)

        model_meta.optimizer_param.CopyFrom(optimizer_param)

        return model_meta

    def set_hetero_nn_model_meta(self, model_meta):
        self.config_type = model_meta.config_type

        if self.config_type == "nn":
            self.bottom_nn_define = []

            for layer in model_meta.bottom_nn_define:
                self.bottom_nn_define.append(json.loads(layer))

        elif self.config_type == 'keras':
            self.bottom_nn_define = json.loads(model_meta.bottom_nn_define[0])

        if self.optimizer is None:
            from types import SimpleNamespace
            self.optimizer = SimpleNamespace(optimizer=None, kwargs={})
            self.optimizer.optimizer = model_meta.optimizer_param.optimizer
            self.optimizer.kwargs = json.loads(model_meta.optimizer_param.kwargs)

    def set_hetero_nn_model_param(self, model_param):
        self.bottom_model_input_shape = model_param.bottom_model_input_shape
        self._restore_bottom_model(model_param.bottom_saved_model_bytes)
        self._restore_interactive_model(model_param.interactive_layer_param)

    def get_hetero_nn_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.interactive_layer_param.CopyFrom(self.interactive_model.export_model())

        return model_param

    def train(self, x, epoch, batch_idx):
        if self.bottom_model is None:
            self.bottom_model_input_shape = x.shape[1]
            self._build_bottom_model()
            self._build_interactive_model()
            if self.batch_size == -1:
                self.batch_size = x.shape[0]

            if self.selector:
                self.bottom_model.set_backward_select_strategy()
                self.bottom_model.set_batch(self.batch_size)
                self.interactive_model.set_backward_select_strategy()

        host_bottom_output = self.bottom_model.forward(x)

        self.interactive_model.forward(host_bottom_output, epoch, batch_idx, train=True)

        host_gradient, selective_ids = self.interactive_model.backward(epoch, batch_idx)

        self.bottom_model.backward(x, host_gradient, selective_ids)

    def predict(self, x):
        guest_bottom_output = self.bottom_model.predict(x)
        self.interactive_model.forward(guest_bottom_output, train=False)

    def evaluate(self, x, epoch, batch_idx):
        guest_bottom_output = self.bottom_model.predict(x)
        self.interactive_model.forward(guest_bottom_output, epoch, batch_idx, train=False)
