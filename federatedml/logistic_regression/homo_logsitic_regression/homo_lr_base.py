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

from arch.api.utils import log_utils
from federatedml.logistic_regression.base_logistic_regression import BaseLogisticRegression
from federatedml.optim.optimizer import optimizer_factory
from federatedml.statistic import data_overview
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.homo_lr_transfer_variable import HomoLRTransferVariable

LOGGER = log_utils.getLogger()


class HomoLRBase(BaseLogisticRegression):
    def __init__(self):
        super(HomoLRBase, self).__init__()
        self.model_name = 'HomoLogisticRegression'
        self.model_param_name = 'HomoLogisticRegressionParam'
        self.model_meta_name = 'HomoLogisticRegressionMeta'
        self.mode = consts.HOMO
        self.aggregator = None

    def _init_model(self, params):
        super(HomoLRBase, self)._init_model(params)

        if params.encrypt_param.method == consts.PAILLIER:
            if params.optimizer != 'sgd':
                raise ValueError("Paillier encryption mode supports 'sgd' optimizer method only.")

        self.transfer_variable = HomoLRTransferVariable()
        self.aggregator.register_aggregator(self.transfer_variable)
        self.aggregator.initialize_aggregator(params.party_weight)
        self.optimizer = optimizer_factory(params)

    @property
    def use_loss(self):
        if self.model_param.converge_func == 'weight_diff':
            return False
        return True

    def _judge_stage(self, args):
        data_sets = args['data']
        has_eval = False
        for data_key in data_sets:
            if "eval_data" in data_sets[data_key]:
                has_eval = True

        if "model" in args:
            stage = 'predict'
        else:
            stage = 'fit'
        LOGGER.debug("Current stage: {}, has_eval: {}".format(stage, has_eval))
        return stage, has_eval

    def _extract_data(self, data_sets):
        train_data = None
        eval_data = None
        for data_key in data_sets:
            if data_sets[data_key].get("train_data", None):
                train_data = data_sets[data_key]["train_data"]

            if data_sets[data_key].get("eval_data", None):
                eval_data = data_sets[data_key]["eval_data"]
        return train_data, eval_data

    def _init_model_variables(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        lr_weights = self.initializer.init_model(model_shape, init_params=self.init_param_obj,
                                                   data_instance=data_instances)
        return lr_weights

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)
        train_data, eval_data = self._extract_data(args["data"])
        stage, has_eval = self._judge_stage(args)
        if self.need_cv:
            LOGGER.info("Need cross validation.")
            self.cross_validation(train_data)

        elif self.need_one_vs_rest:
            if "model" in args:
                self._load_model(args)
            self.one_vs_rest_logic(stage, train_data, eval_data)

        elif stage == "fit":
            self.fit(train_data)
            self.data_output = self.predict(train_data)
            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda value: value + ["train"])
            if has_eval:
                self.set_flowid('validate')
                eval_data_output = self.predict(eval_data)
                if eval_data_output:
                    eval_data_output = eval_data_output.mapValues(lambda value: value + ["validation"])
                    self.data_output = self.data_output.union(eval_data_output)
            if train_data is not None:
                self.set_predict_data_schema(self.data_output, train_data.schema)
        else:
            self.set_flowid('predict')
            self.data_output = self.predict(eval_data)
            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda value: value + ["test"])

            if eval_data is not None:
                self.set_predict_data_schema(self.data_output, eval_data.schema)
