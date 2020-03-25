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
import numpy as np

from arch.api.utils import log_utils
from federatedrec.factorization_machine.base_factorization_machine import BaseFactorizationMachine
from federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights
from federatedml.optim import activation
from federatedml.optim.optimizer import optimizer_factory
from federatedrec.param.factorization_machine_param import HomoFactorizationParam
from federatedml.protobuf.generated import fm_model_meta_pb2
from federatedml.secureprotol import FakeEncrypt
from federatedml.statistic import data_overview
from federatedrec.transfer_variable.transfer_class.homo_fm_transfer_variable import HomoFMTransferVariable
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HomoFMBase(BaseFactorizationMachine):
    def __init__(self):
        super(HomoFMBase, self).__init__()
        self.model_name = 'HomoFactorizationMachine'
        self.model_param_name = 'HomoFactorizationMachineParam'
        self.model_meta_name = 'HomoFactorizationMachineMeta'
        self.mode = consts.HOMO
        self.model_param = HomoFactorizationParam()
        self.aggregator = None

    def _init_model(self, params):
        super(HomoFMBase, self)._init_model(params)

        # if params.encrypt_param.method == consts.PAILLIER:
        #     self.cipher_operator = PaillierEncrypt()
        # else:
        #     self.cipher_operator = FakeEncrypt()

        self.transfer_variable = HomoFMTransferVariable()
        self.aggregator.register_aggregator(self.transfer_variable)
        self.optimizer = optimizer_factory(params)
        self.aggregate_iters = params.aggregate_iters

    @property
    def use_loss(self):
        if self.model_param.early_stop == 'weight_diff':
            return False
        return True

    def compute_wx_plus_fm(self, data_instances, model_weights):
        """ calculate w*x + (v*x)^2 - (v^2)*(x^2)
            data_instances: batch_size * feature_size
            w_: feature_size
            embed_: feature_size * embed_size
        """

        def fm_func(features, embed):
            re = np.multiply(np.expand_dims(features, 1), embed)
            re = np.sum(re, 0)
            part1 = np.sum(np.power(re, 2))
            features_square = np.power(features, 2)
            embed_square = np.power(embed, 2)
            part2 = np.sum(np.dot(features_square, embed_square))
            return 0.5 * (part1 - part2)

        wx = data_instances.mapValues(lambda v: np.dot(v.features, model_weights.w_) + model_weights.intercept_)
        LOGGER.info("wx:{}".format(wx.take(20)))
        fm = data_instances.mapValues(lambda v: fm_func(v.features, model_weights.embed_))
        LOGGER.info("fm:{}".format(fm.take(20)))
        return wx.join(fm, lambda wx_, fm_: wx_ + fm_)

    def compute_vx(self, data_instances, embed_):
        return data_instances.mapValues(lambda  v: np.dot(v.features, embed_))

    def classify(self, predict_wx, threshold):
        """
        convert a probability table into a predicted class table.
        """
        # predict_wx = self.compute_wx(data_instances, self.model_weights.w_, self.model_weights.intercept_)

        def predict(x):
            prob = activation.sigmoid(x)
            pred_label = 1 if prob > threshold else 0
            return prob, pred_label

        predict_table = predict_wx.mapValues(predict)
        return predict_table

    def _init_model_variables(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        fit_intercept = False
        if self.init_param_obj.fit_intercept:
            fit_intercept = True
            self.init_param_obj.fit_intercept = False
        w_ = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        embed_ = self.initializer.init_model([model_shape, self.init_param_obj.embed_size],
                                             init_params=self.init_param_obj)
        model_weights = \
            FactorizationMachineWeights(w_, embed_, fit_intercept=fit_intercept)
        return model_weights

    def _compute_loss(self, data_instances):
        f = functools.partial(self.gradient_operator.compute_loss,
                              w=self.model_weights.w_,
                              embed=self.model_weights.embed_,
                              intercept=self.model_weights.intercept_)
        loss = data_instances.mapPartitions(f).reduce(fate_operator.reduce_add)
        loss_norm = self.optimizer.loss_norm(self.model_weights)
        if loss_norm is not None:
            loss += loss_norm
        loss /= data_instances.count()
        self.callback_loss(self.n_iter_, loss)
        self.loss_history.append(loss)
        return loss

    def _get_meta(self):
        meta_protobuf_obj = fm_model_meta_pb2.FMModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          need_one_vs_rest=self.need_one_vs_rest)
        return meta_protobuf_obj
