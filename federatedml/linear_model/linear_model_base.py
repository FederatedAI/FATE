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

from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.model_selection import start_cross_validation
from federatedml.model_selection.stepwise import start_stepwise
from federatedml.optim.convergence import converge_func_factory
from federatedml.optim.initialize import Initializer
from federatedml.optim.optimizer import optimizer_factory
from federatedml.statistic import data_overview
from federatedml.util import abnormal_detection
from federatedml.util.validation_strategy import ValidationStrategy


class BaseLinearModel(ModelBase):
    def __init__(self):
        super(BaseLinearModel, self).__init__()
        # attribute:
        self.n_iter_ = 0
        self.classes_ = None
        self.feature_shape = None
        self.gradient_operator = None
        self.initializer = Initializer()
        self.transfer_variable = None
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.model_name = 'toSet'
        self.model_param_name = 'toSet'
        self.model_meta_name = 'toSet'
        self.role = ''
        self.mode = ''
        self.schema = {}
        self.cipher_operator = None
        self.model_weights = None
        self.validation_freqs = None
        self.need_one_vs_rest = False
        self.need_call_back_loss = True
        self.init_param_obj = None

    def _init_model(self, params):
        self.model_param = params
        self.alpha = params.alpha
        self.init_param_obj = params.init_param
        # self.fit_intercept = self.init_param_obj.fit_intercept
        self.batch_size = params.batch_size
        self.max_iter = params.max_iter
        self.optimizer = optimizer_factory(params)
        self.converge_func = converge_func_factory(params.early_stop, params.tol)
        self.encrypted_calculator = None
        self.validation_freqs = params.validation_freqs
        self.validation_strategy = None
        self.early_stopping_rounds = params.early_stopping_rounds

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def set_header(self, header):
        self.header = header

    def get_header(self, data_instances):
        if self.header is not None:
            return self.header
        return data_instances.schema.get("header")

    @property
    def fit_intercept(self):
        return self.init_param_obj.fit_intercept

    def _get_meta(self):
        raise NotImplementedError("This method should be be called here")

    def _get_param(self):
        raise NotImplementedError("This method should be be called here")

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def disable_callback_loss(self):
        self.need_call_back_loss = False

    def enable_callback_loss(self):
        self.need_call_back_loss = True

    def callback_loss(self, iter_num, loss):
        metric_meta = MetricMeta(name='train',
                                 metric_type="LOSS",
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def init_validation_strategy(self, train_data=None, validate_data=None):
        validation_strategy = ValidationStrategy(self.role, self.mode, self.validation_freqs, self.early_stopping_rounds)
        validation_strategy.set_train_data(train_data)
        validation_strategy.set_validate_data(validate_data)
        return validation_strategy

    def cross_validation(self, data_instances):
        return start_cross_validation.run(self, data_instances)

    def stepwise(self, data_instances):
        self.disable_callback_loss()
        return start_stepwise.run(self, data_instances)

    def _get_cv_param(self):
        self.model_param.cv_param.role = self.role
        self.model_param.cv_param.mode = self.mode
        return self.model_param.cv_param

    def _get_stepwise_param(self):
        self.model_param.stepwise_param.role = self.role
        self.model_param.stepwise_param.mode = self.mode
        return self.model_param.stepwise_param

    def set_schema(self, data_instance, header=None):
        if header is None:
            self.schema["header"] = self.header
        else:
            self.schema["header"] = header
        data_instance.schema = self.schema
        return data_instance

    def init_schema(self, data_instance):
        if data_instance is None:
            return
        self.schema = data_instance.schema
        self.header = self.schema.get('header')
