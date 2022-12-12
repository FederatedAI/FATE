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
from federatedml.linear_model.coordinated_linear_model.\
    logistic_regression.homo_logistic_regression.homo_lr_base import HomoLRBase
from federatedml.nn.homo.trainer.fedavg_trainer import FedAVGTrainer
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.model_base import MetricMeta
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.callbacks.validation_strategy import ValidationStrategy
from federatedml.nn.homo.trainer.trainer_base import ExporterBase
from federatedml.protobuf.generated import lr_model_param_pb2, lr_model_meta_pb2


class HomoLRServerExporter(ExporterBase):

    def __init__(self, param_name, meta_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_name = param_name
        self.meta_name = meta_name

    def export_model_dict(self, *args, **kwargs):
        # return empty model only
        return {self.param_name: lr_model_param_pb2.LRModelParam(), self.meta_name: lr_model_meta_pb2.LRModelMeta()}


class HomoLRServer(HomoLRBase):
    def __init__(self):
        super(HomoLRServer, self).__init__()
        self.re_encrypt_times = []  # Record the times needed for each host
        self.role = consts.ARBITER
        self.trainer = None

        # check point
        self.save_freq = None
        self.model_checkpoint = None

    def _init_model(self, params):
        super()._init_model(params)

    def fit_binary(self, data_instances=None, validate_data=None):

        for callback_cpn in self.callback_list.callback_list:
            if isinstance(callback_cpn, ModelCheckpoint):
                self.save_freq = callback_cpn.save_freq
                self.model_checkpoint = callback_cpn
            elif isinstance(callback_cpn, ValidationStrategy):
                self.validation_freqs = callback_cpn.validation_freqs

        # fate loss callback setting
        if not self.callback_one_vs_rest:  # ovr does not display loss
            self.callback_meta(
                "loss", "train", MetricMeta(
                    name="train", metric_type="LOSS", extra_metas={
                        "unit_name": "aggregate_round"}))

        early_stop = None
        if self.early_stop != 'weight_diff':
            early_stop = self.early_stop
        self.trainer = FedAVGTrainer(
            epochs=self.max_iter,
            secure_aggregate=True,
            aggregate_every_n_epoch=self.aggregate_iters,
            validation_freqs=self.validation_freqs,
            task_type='binary',
            checkpoint_save_freqs=self.save_freq,
            early_stop=early_stop,
            tol=self.tol,
            shuffle=False
        )
        if self.one_vs_rest_obj is None:
            self.trainer.set_tracker(self.tracker)
        self.trainer.set_checkpoint(self.model_checkpoint)
        self.trainer.set_model_exporter(HomoLRServerExporter(self.model_param_name, self.model_meta_name))

        self.trainer.server_aggregate_procedure()
        LOGGER.info("Finish Training task")

    def predict(self, data_instantces=None):
        LOGGER.info(f'Start predict task')
        pass

    def export_model(self):
        # arbiter does not save models
        return None

    def load_model(self, model_dict):
        # do nothing now
        return None
