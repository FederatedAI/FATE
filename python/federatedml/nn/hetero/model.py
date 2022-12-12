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

import json
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.param.hetero_nn_param import HeteroNNParam
from federatedml.nn.hetero.strategy.selector import SelectorFactory
from federatedml.nn.hetero.nn_component.bottom_model import BottomModel
from federatedml.nn.hetero.nn_component.top_model import TopModel
from federatedml.nn.backend.utils.common import global_seed
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNModelMeta
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import OptimizerParam
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNModelParam
from federatedml.nn.hetero.interactive.he_interactive_layer import HEInteractiveLayerGuest, HEInteractiveLayerHost


class HeteroNNModel(object):
    def __init__(self):
        self.partition = 1
        self.batch_size = None
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None
        self.optimizer = None
        self.config_type = None
        self.transfer_variable = None

        self._predict_round = 0

    def load_model(self):
        pass

    def predict(self, data):
        pass

    def export_model(self):
        pass

    def get_hetero_nn_model_meta(self):
        pass

    def get_hetero_nn_model_param(self):
        pass

    def set_hetero_nn_model_meta(self, model_meta):
        pass

    def set_hetero_nn_model_param(self, model_param):
        pass

    def set_partition(self, partition):
        pass

    def inc_predict_round(self):
        self._predict_round += 1


class HeteroNNGuestModel(HeteroNNModel):

    def __init__(self, hetero_nn_param, component_properties, flowid):
        super(HeteroNNGuestModel, self).__init__()

        self.role = consts.GUEST
        self.bottom_model: BottomModel = None
        self.top_model: TopModel = None
        self.interactive_model: HEInteractiveLayerGuest = None
        self.loss = None
        self.hetero_nn_param = None
        self.is_empty = False
        self.coae_param = None
        self.seed = 100
        self.set_nn_meta(hetero_nn_param)
        self.component_properties = component_properties
        self.flowid = flowid
        self.label_num = 1
        self.selector = SelectorFactory.get_selector(
            hetero_nn_param.selector_param.method,
            hetero_nn_param.selector_param.selective_size,
            beta=hetero_nn_param.selector_param.beta,
            random_rate=hetero_nn_param.selector_param.random_state,
            min_prob=hetero_nn_param.selector_param.min_prob)

    def set_nn_meta(self, hetero_nn_param: HeteroNNParam):
        self.bottom_nn_define = hetero_nn_param.bottom_nn_define
        self.top_nn_define = hetero_nn_param.top_nn_define
        self.interactive_layer_define = hetero_nn_param.interactive_layer_define
        self.config_type = hetero_nn_param.config_type
        self.optimizer = hetero_nn_param.optimizer
        self.loss = hetero_nn_param.loss
        self.hetero_nn_param = hetero_nn_param
        self.batch_size = hetero_nn_param.batch_size
        self.seed = hetero_nn_param.seed

        coae_param = hetero_nn_param.coae_param
        if coae_param.enable:
            self.coae_param = coae_param

    def set_empty(self):
        self.is_empty = True

    def set_label_num(self, label_num):
        self.label_num = label_num
        if self.top_model is not None:  # warmstart case
            self.top_model.label_num = label_num

    def train(self, x, y, epoch, batch_idx):

        if self.batch_size == -1:
            self.batch_size = x.shape[0]

        global_seed(self.seed)

        if self.top_model is None:
            self._build_top_model()
            LOGGER.debug('top model is {}'.format(self.top_model))

        if not self.is_empty:
            if self.bottom_model is None:
                self._build_bottom_model()
                LOGGER.debug('bottom model is {}'.format(self.bottom_model))
            self.bottom_model.train_mode(True)
            guest_bottom_output = self.bottom_model.forward(x)
        else:
            guest_bottom_output = None

        if self.interactive_model is None:
            self._build_interactive_model()

        interactive_output = self.interactive_model.forward(
            x=guest_bottom_output, epoch=epoch, batch=batch_idx, train=True)
        self.top_model.train_mode(True)
        selective_ids, gradients, loss = self.top_model.train_and_get_backward_gradient(
            interactive_output, y)
        interactive_layer_backward = self.interactive_model.backward(
            error=gradients, epoch=epoch, batch=batch_idx, selective_ids=selective_ids)

        if not self.is_empty:
            self.bottom_model.backward(
                x, interactive_layer_backward, selective_ids)

        return loss

    def predict(self, x, batch=0):

        if not self.is_empty:
            self.bottom_model.train_mode(False)
            guest_bottom_output = self.bottom_model.predict(x)
        else:
            guest_bottom_output = None

        interactive_output = self.interactive_model.forward(
            guest_bottom_output, epoch=self._predict_round, batch=batch, train=False)

        self.top_model.train_mode(False)
        preds = self.top_model.predict(interactive_output)
        # prediction procedure has its prediction iteration count, we do this
        # to avoid reusing communication suffixes
        self.inc_predict_round()

        return preds

    def get_hetero_nn_model_param(self):

        model_param = HeteroNNModelParam()
        model_param.is_empty = self.is_empty
        if not self.is_empty:
            model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.top_saved_model_bytes = self.top_model.export_model()
        model_param.interactive_layer_param.CopyFrom(
            self.interactive_model.export_model())
        coae_bytes = self.top_model.export_coae()
        if coae_bytes is not None:
            model_param.coae_bytes = coae_bytes

        return model_param

    def set_hetero_nn_model_param(self, model_param):

        self.is_empty = model_param.is_empty
        if not self.is_empty:
            self._restore_bottom_model(model_param.bottom_saved_model_bytes)
        self._restore_interactive_model(model_param.interactive_layer_param)
        self._restore_top_model(model_param.top_saved_model_bytes)
        self.top_model.restore_coae(model_param.coae_bytes)

    def get_hetero_nn_model_meta(self):

        model_meta = HeteroNNModelMeta()
        model_meta.config_type = self.config_type
        model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))
        model_meta.top_nn_define.append(json.dumps(self.top_nn_define))
        model_meta.interactive_layer_define = json.dumps(
            self.interactive_layer_define)
        model_meta.interactive_layer_lr = self.hetero_nn_param.interactive_layer_lr
        optimizer_param = OptimizerParam()
        model_meta.loss = json.dumps(self.loss)
        optimizer_param.optimizer = self.optimizer['optimizer']
        tmp_dict = copy.deepcopy(self.optimizer)
        tmp_dict.pop('optimizer')
        optimizer_param.kwargs = json.dumps(tmp_dict)
        model_meta.optimizer_param.CopyFrom(optimizer_param)

        return model_meta

    def set_hetero_nn_model_meta(self, model_meta):
        self.config_type = model_meta.config_type
        self.bottom_nn_define = json.loads(model_meta.bottom_nn_define[0])
        self.top_nn_define = json.loads(model_meta.top_nn_define[0])
        self.interactive_layer_define = json.loads(
            model_meta.interactive_layer_define)
        self.loss = json.loads(model_meta.loss)

        if self.optimizer is None:
            from types import SimpleNamespace
            self.optimizer = SimpleNamespace(optimizer=None, kwargs={})
            self.optimizer.optimizer = model_meta.optimizer_param.optimizer
            self.optimizer.kwargs = json.loads(
                model_meta.optimizer_param.kwargs)
            tmp_opt = {'optimizer': self.optimizer.optimizer}
            tmp_opt.update(self.optimizer.kwargs)
            self.optimizer = tmp_opt

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition=1):
        self.partition = partition
        if self.interactive_model is not None:
            self.interactive_model.set_partition(self.partition)

    def _init_bottom_select_strategy(self):
        if self.selector:
            self.bottom_model.set_backward_select_strategy()
            self.bottom_model.set_batch(self.batch_size)

    def _build_bottom_model(self):

        self.bottom_model = BottomModel(
            optimizer=self.optimizer,
            layer_config=self.bottom_nn_define)
        self._init_bottom_select_strategy()

    def _restore_bottom_model(self, model_bytes):
        self._build_bottom_model()
        self.bottom_model.restore_model(model_bytes)
        self._init_bottom_select_strategy()

    def _init_top_select_strategy(self):
        if self.selector:
            self.top_model.set_backward_selector_strategy(
                selector=self.selector)
            self.top_model.set_batch(self.batch_size)

    def _build_top_model(self):
        if self.top_nn_define is None:
            raise ValueError(
                'top nn model define is None, you must define your top model in guest side')
        self.top_model = TopModel(
            optimizer=self.optimizer,
            layer_config=self.top_nn_define,
            loss=self.loss,
            coae_config=self.coae_param,
            label_num=self.label_num
        )

        self._init_top_select_strategy()

    def _restore_top_model(self, model_bytes):
        self._build_top_model()
        self.top_model.restore_model(model_bytes)
        self._init_top_select_strategy()

    def _init_inter_layer(self):
        self.interactive_model.set_partition(self.partition)
        self.interactive_model.set_batch(self.batch_size)
        self.interactive_model.set_flow_id('{}_interactive_layer'.format(self.flowid))
        if self.selector:
            self.interactive_model.set_backward_select_strategy()

    def _build_interactive_model(self):
        self.interactive_model = HEInteractiveLayerGuest(
            params=self.hetero_nn_param,
            layer_config=self.interactive_layer_define,
            host_num=len(
                self.component_properties.host_party_idlist))
        self._init_inter_layer()

    def _restore_interactive_model(self, interactive_model_param):
        self._build_interactive_model()
        self.interactive_model.restore_model(interactive_model_param)
        self._init_inter_layer()


class HeteroNNHostModel(HeteroNNModel):

    def __init__(self, hetero_nn_param, flowid):
        super(HeteroNNHostModel, self).__init__()

        self.role = consts.HOST
        self.bottom_model: BottomModel = None
        self.interactive_model = None
        self.hetero_nn_param = None
        self.seed = 100
        self.set_nn_meta(hetero_nn_param)
        self.selector = SelectorFactory.get_selector(
            hetero_nn_param.selector_param.method,
            hetero_nn_param.selector_param.selective_size,
            beta=hetero_nn_param.selector_param.beta,
            random_rate=hetero_nn_param.selector_param.random_state,
            min_prob=hetero_nn_param.selector_param.min_prob)
        self.flowid = flowid

    def set_nn_meta(self, hetero_nn_param):
        self.bottom_nn_define = hetero_nn_param.bottom_nn_define
        self.config_type = hetero_nn_param.config_type
        self.optimizer = hetero_nn_param.optimizer
        self.hetero_nn_param = hetero_nn_param
        self.batch_size = hetero_nn_param.batch_size
        self.seed = hetero_nn_param.seed

    def _build_bottom_model(self):
        if self.bottom_nn_define is None:
            raise ValueError(
                'bottom nn model define is None, you must define your bottom model in host')
        self.bottom_model = BottomModel(
            optimizer=self.optimizer,
            layer_config=self.bottom_nn_define)

    def _restore_bottom_model(self, model_bytes):
        self._build_bottom_model()
        self.bottom_model.restore_model(model_bytes)

    def _build_interactive_model(self):
        self.interactive_model = HEInteractiveLayerHost(self.hetero_nn_param)
        self.interactive_model.set_partition(self.partition)
        self.interactive_model.set_flow_id('{}_interactive_layer'.format(self.flowid))

    def _restore_interactive_model(self, interactive_layer_param):
        self._build_interactive_model()
        self.interactive_model.restore_model(interactive_layer_param)
        self.interactive_model.set_partition(self.partition)
        self.interactive_model.set_flow_id('{}_interactive_layer'.format(self.flowid))

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition=1):
        self.partition = partition
        if self.interactive_model is not None:
            self.interactive_model.set_partition(self.partition)

        LOGGER.debug(
            "set_partition, partition num is {}".format(
                self.partition))

    def get_hetero_nn_model_meta(self):
        model_meta = HeteroNNModelMeta()
        model_meta.config_type = self.config_type
        model_meta.bottom_nn_define.append(json.dumps(self.bottom_nn_define))
        model_meta.interactive_layer_lr = self.hetero_nn_param.interactive_layer_lr
        optimizer_param = OptimizerParam()
        optimizer_param.optimizer = self.optimizer['optimizer']
        tmp_opt = copy.deepcopy(self.optimizer)
        tmp_opt.pop('optimizer')
        optimizer_param.kwargs = json.dumps(tmp_opt)
        model_meta.optimizer_param.CopyFrom(optimizer_param)

        return model_meta

    def set_hetero_nn_model_meta(self, model_meta):

        self.config_type = model_meta.config_type
        self.bottom_nn_define = json.loads(model_meta.bottom_nn_define[0])
        if self.optimizer is None:
            from types import SimpleNamespace
            self.optimizer = SimpleNamespace(optimizer=None, kwargs={})
            self.optimizer.optimizer = model_meta.optimizer_param.optimizer
            self.optimizer.kwargs = json.loads(
                model_meta.optimizer_param.kwargs)
            tmp_opt = {'optimizer': self.optimizer.optimizer}
            tmp_opt.update(self.optimizer.kwargs)
            self.optimizer = tmp_opt

    def set_hetero_nn_model_param(self, model_param):
        self._restore_bottom_model(model_param.bottom_saved_model_bytes)
        self._restore_interactive_model(model_param.interactive_layer_param)

    def get_hetero_nn_model_param(self):
        model_param = HeteroNNModelParam()
        model_param.bottom_saved_model_bytes = self.bottom_model.export_model()
        model_param.interactive_layer_param.CopyFrom(
            self.interactive_model.export_model())

        return model_param

    def train(self, x, epoch, batch_idx):

        if self.bottom_model is None:
            global_seed(self.seed)
            self._build_bottom_model()
            if self.batch_size == -1:
                self.batch_size = x.shape[0]
            self._build_interactive_model()
            if self.selector:
                self.bottom_model.set_backward_select_strategy()
                self.bottom_model.set_batch(self.batch_size)
                self.interactive_model.set_backward_select_strategy()

        self.bottom_model.train_mode(True)
        host_bottom_output = self.bottom_model.forward(x)

        self.interactive_model.forward(
            host_bottom_output, epoch, batch_idx, train=True)

        host_gradient, selective_ids = self.interactive_model.backward(
            epoch, batch_idx)

        self.bottom_model.backward(x, host_gradient, selective_ids)

    def predict(self, x, batch=0):
        self.bottom_model.train_mode(False)
        guest_bottom_output = self.bottom_model.predict(x)
        self.interactive_model.forward(
            guest_bottom_output,
            epoch=self._predict_round,
            batch=batch,
            train=False)
        # prediction procedure has its prediction iteration count, we do this
        # to avoid reusing communication suffixes
        self.inc_predict_round()
