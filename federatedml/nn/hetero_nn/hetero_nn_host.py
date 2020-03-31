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


import numpy as np

from arch.api.utils import log_utils
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.nn.hetero_nn.backend.model_builder import model_builder
from federatedml.nn.hetero_nn.hetero_nn_base import HeteroNNBase
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNMeta
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNParam

LOGGER = log_utils.getLogger()
MODELMETA = "HeteroNNHostMeta"
MODELPARAM = "HeteroNNHostParam"


class HeteroNNHost(HeteroNNBase):
    def __init__(self):
        super(HeteroNNHost, self).__init__()

        self.batch_generator = batch_generator.Host()
        self.model = None

        self.input_shape = None
        self.validation_strategy = None

    def _init_model(self, hetero_nn_param):
        super(HeteroNNHost, self)._init_model(hetero_nn_param)

    def export_model(self):
        if self.model is None:
            return

        return {MODELMETA: self._get_model_meta(),
                MODELPARAM: self._get_model_param()}

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        param = model_dict.get(MODELPARAM)
        meta = model_dict.get(MODELMETA)

        self._build_model()
        self._restore_model_meta(meta)
        self._restore_model_param(param)

    def _build_model(self):
        self.model = model_builder("host", self.hetero_nn_param)
        self.model.set_transfer_variable(self.transfer_variable)

    def predict(self, data_inst):
        test_x = self._load_data(data_inst)
        self.set_partition(data_inst)

        self.model.predict(test_x)

    def fit(self, data_inst, validate_data=None):
        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)
        self._build_model()
        self.prepare_batch_data(self.batch_generator, data_inst)

        cur_epoch = 0

        while cur_epoch < self.epochs:
            for batch_idx in range(len(self.data_x)):
                self.model.train(self.data_x[batch_idx], cur_epoch, batch_idx)

                self.reset_flowid()
                self.model.evaluate(self.data_x[batch_idx], cur_epoch, batch_idx)
                self.recovery_flowid()

            if self.validation_strategy:
                self.validation_strategy.validate(self, cur_epoch)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            is_converge = self.transfer_variable.is_converge.get(idx=0,
                                                                 suffix=(cur_epoch,))

            if is_converge:
                LOGGER.debug("Training process is converged in epoch {}".format(cur_epoch))
                break

            cur_epoch += 1

        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)

    def prepare_batch_data(self, batch_generator, data_inst):
        batch_generator.initialize_batch_generator(data_inst)
        batch_data_generator = batch_generator.generate_batch_data()

        for batch_data in batch_data_generator:
            batch_x = self._load_data(batch_data)
            self.data_x.append(batch_x)

        self.set_partition(data_inst)

    def _load_data(self, data_inst):
        data = list(data_inst.collect())
        data_keys = [key for (key, val) in data]
        data_keys_map = dict(zip(sorted(data_keys), range(len(data_keys))))
        batch_x = [None for i in range(len(data_keys))]

        for key, inst in data:
            batch_x[data_keys_map[key]] = inst.features

            if self.input_shape is None:
                self.input_shape = inst.features.shape

        batch_x = np.asarray(batch_x)

        return batch_x

    def _get_model_meta(self):
        model_meta = HeteroNNMeta()
        model_meta.batch_size = self.batch_size
        model_meta.hetero_nn_model_meta.CopyFrom(self.model.get_hetero_nn_model_meta())

        return model_meta

    def _get_model_param(self):
        model_param = HeteroNNParam()
        model_param.hetero_nn_model_param.CopyFrom(self.model.get_hetero_nn_model_param())

        return model_param
