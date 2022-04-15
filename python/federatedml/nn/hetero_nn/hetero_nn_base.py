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
#
from federatedml.model_base import ModelBase
from federatedml.model_selection import start_cross_validation
from federatedml.param.hetero_nn_param import HeteroNNParam
from federatedml.transfer_variable.transfer_class.hetero_nn_transfer_variable import HeteroNNTransferVariable
from federatedml.util import consts


class HeteroNNBase(ModelBase):
    def __init__(self):
        super(HeteroNNBase, self).__init__()

        self.tol = None
        self.early_stop = None

        self.epochs = None
        self.batch_size = None
        self._header = []

        self.predict_param = None
        self.hetero_nn_param = None

        self.model_builder = None

        self.batch_generator = None
        self.model = None

        self.partition = None
        self.validation_freqs = None
        self.early_stopping_rounds = None
        self.metrics = []
        self.use_first_metric_only = False

        self.data_x = []
        self.data_y = []
        self.transfer_variable = HeteroNNTransferVariable()
        self.model_param = HeteroNNParam()

        self.mode = consts.HETERO

        self.selector_param = None

        self.floating_point_precision = None

        self.history_iter_epoch = 0
        self.iter_epoch = 0

    def _init_model(self, hetero_nn_param):
        self.interactive_layer_lr = hetero_nn_param.interactive_layer_lr
        self.epochs = hetero_nn_param.epochs
        self.batch_size = hetero_nn_param.batch_size

        self.early_stop = hetero_nn_param.early_stop
        self.validation_freqs = hetero_nn_param.validation_freqs
        self.early_stopping_rounds = hetero_nn_param.early_stopping_rounds
        self.metrics = hetero_nn_param.metrics
        self.use_first_metric_only = hetero_nn_param.use_first_metric_only

        self.tol = hetero_nn_param.tol

        self.predict_param = hetero_nn_param.predict_param
        self.hetero_nn_param = hetero_nn_param

        self.selector_param = hetero_nn_param.selector_param

        self.floating_point_precision = hetero_nn_param.floating_point_precision

        if self.role == consts.GUEST:
            self.batch_generator.register_batch_generator(self.transfer_variable, has_arbiter=False)
        else:
            self.batch_generator.register_batch_generator(self.transfer_variable)

    def reset_flowid(self):
        new_flowid = ".".join([self.flowid, "evaluate"])
        self.set_flowid(new_flowid)

    def recovery_flowid(self):
        new_flowid = ".".join(self.flowid.split(".", -1)[: -1])
        self.set_flowid(new_flowid)

    def _build_bottom_model(self):
        pass

    def _build_interactive_model(self):
        pass

    def prepare_batch_data(self, batch_generator, data_inst):
        pass

    def _load_data(self, data_inst):
        pass

    def _restore_model_meta(self, meta):
        # self.hetero_nn_param.interactive_layer_lr = meta.interactive_layer_lr
        self.hetero_nn_param.task_type = meta.task_type
        if not self.component_properties.is_warm_start:
            self.batch_size = meta.batch_size
            self.epochs = meta.epochs
            self.tol = meta.tol
            self.early_stop = meta.early_stop

        self.model.set_hetero_nn_model_meta(meta.hetero_nn_model_meta)

    def _restore_model_param(self, param):
        self.model.set_hetero_nn_model_param(param.hetero_nn_model_param)
        self._header = list(param.header)
        self.history_iter_epoch = param.iter_epoch
        self.iter_epoch = param.iter_epoch

    def set_partition(self, data_inst):
        self.partition = data_inst.partitions
        self.model.set_partition(self.partition)

    def cross_validation(self, data_instances):
        return start_cross_validation.run(self, data_instances)
