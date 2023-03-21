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

import functools
import torch as t
import math
from torch.optim.lr_scheduler import LambdaLR
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.coordinated_linear_model.logistic_regression.base_logistic_regression import BaseLogisticRegression
from federatedml.optim import activation
from federatedml.param.logistic_regression_param import HomoLogisticParam
from federatedml.protobuf.generated import lr_model_meta_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.statistic import data_overview
from federatedml.transfer_variable.transfer_class.homo_lr_transfer_variable import HomoLRTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator


class HomoLRBase(BaseLogisticRegression):
    def __init__(self):
        super(HomoLRBase, self).__init__()
        self.model_name = 'HomoLogisticRegression'
        self.model_param_name = 'HomoLogisticRegressionParam'
        self.model_meta_name = 'HomoLogisticRegressionMeta'
        self.mode = consts.HOMO
        self.model_param = HomoLogisticParam()
        self.aggregator = None
        self.param = None

    def get_torch_optimizer(self, torch_model: t.nn.Module, param: HomoLogisticParam):

        try:
            learning_rate = param.learning_rate
            alpha = param.alpha  # L2 penalty weight

            decay = param.decay
            decay_sqrt = param.decay_sqrt

            if not decay_sqrt:
                def decay_func(epoch): return 1 / (1 + epoch * decay)
            else:
                def decay_func(epoch): return 1 / math.sqrt(1 + epoch * decay)

        except AttributeError:
            raise AttributeError("Optimizer parameters has not been totally set")

        optimizer_type = param.optimizer
        if optimizer_type == 'sgd':
            opt = t.optim.SGD(params=torch_model.parameters(), lr=learning_rate, weight_decay=alpha)
        elif optimizer_type == 'nesterov_momentum_sgd':
            opt = t.optim.SGD(
                params=torch_model.parameters(),
                nesterov=True,
                momentum=0.9,
                lr=learning_rate,
                weight_decay=alpha)
        elif optimizer_type == 'rmsprop':
            opt = t.optim.RMSprop(params=torch_model.parameters(), alpha=0.99, lr=learning_rate, weight_decay=alpha)
        elif optimizer_type == 'adam':
            opt = t.optim.Adam(params=torch_model.parameters(), lr=learning_rate, weight_decay=alpha)
        elif optimizer_type == 'adagrad':
            opt = t.optim.Adagrad(params=torch_model.parameters(), lr=learning_rate, weight_decay=alpha)
        else:
            if optimizer_type == 'sqn':
                raise NotImplementedError("Sqn optimizer is not supported in Homo-LR")
            raise NotImplementedError("Optimize method cannot be recognized: {}".format(optimizer_type))

        scheduler = LambdaLR(opt, lr_lambda=decay_func)
        return opt, scheduler

    def _init_model(self, params):
        super(HomoLRBase, self)._init_model(params)

        self.transfer_variable = HomoLRTransferVariable()
        # self.aggregator.register_aggregator(self.transfer_variable)
        self.param = params
        self.aggregate_iters = params.aggregate_iters

    @property
    def use_loss(self):
        if self.model_param.early_stop == 'weight_diff':
            return False
        return True

    def fit(self, data_instances, validate_data=None):
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if self.role == consts.ARBITER:
            self._server_check_data()
        else:
            self._client_check_data(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
            if self.header is None:
                self.header = self.one_vs_rest_obj.header
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data):
        raise NotImplementedError("Should not called here")

    def _client_check_data(self, data_instances):
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.init_schema(data_instances)

        # Support multi-class now
        """
        num_classes, classes_ = ClassifyLabelChecker.validate_label(data_instances)
        aligned_label, new_label_mapping = HomoLabelEncoderClient().label_alignment(classes_)
        if len(aligned_label) > 2:
            raise ValueError("Homo LR support binary classification only now")
        elif len(aligned_label) <= 1:
            raise ValueError("Number of classes should be equal to 2")
        """

    def _server_check_data(self):
        # HomoLabelEncoderArbiter().label_alignment()
        pass

    def classify(self, predict_wx, threshold):
        """
        convert a probability table into a predicted class table.
        """

        # predict_wx = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)

        def predict(x):
            prob = activation.sigmoid(x)
            pred_label = 1 if prob > threshold else 0
            return prob, pred_label

        predict_table = predict_wx.mapValues(predict)
        return predict_table

    def _init_model_variables(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj,
                                        data_instance=data_instances)
        model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept)
        return model_weights

    def _compute_loss(self, data_instances, prev_round_weights):
        f = functools.partial(self.gradient_operator.compute_loss,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        loss = data_instances.applyPartitions(f).reduce(fate_operator.reduce_add)
        if self.use_proximal:  # use additional proximal term
            loss_norm = self.optimizer.loss_norm(self.model_weights,
                                                 prev_round_weights)
        else:
            loss_norm = self.optimizer.loss_norm(self.model_weights)

        if loss_norm is not None:
            loss += loss_norm
        loss /= data_instances.count()
        if self.need_call_back_loss:
            self.callback_loss(self.n_iter_, loss)
        self.loss_history.append(loss)
        return loss

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          module='HomoLR',
                                                          need_one_vs_rest=self.need_one_vs_rest)
        return meta_protobuf_obj
