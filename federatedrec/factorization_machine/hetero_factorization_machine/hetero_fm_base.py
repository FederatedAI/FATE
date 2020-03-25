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

import numpy as np

from federatedml.util import consts
from federatedml.secureprotol import PaillierEncrypt
from federatedml.protobuf.generated import fm_model_meta_pb2
from federatedrec.param.factorization_machine_param import HeteroFactorizationParam
from federatedrec.factorization_machine.base_factorization_machine import BaseFactorizationMachine
from federatedrec.transfer_variable.transfer_class.hetero_fm_transfer_variable import HeteroFMTransferVariable


class HeteroFMBase(BaseFactorizationMachine):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroFactorizationMachine'
        self.model_param_name = 'HeteroFactorizationMachineParam'
        self.model_meta_name = 'HeteroFactorizationMachineMeta'
        self.mode = consts.HETERO
        self.aggregator = None
        self.cipher = None
        self.batch_generator = None
        self.gradient_loss_operator = None
        self.converge_procedure = None
        self.model_param = HeteroFactorizationParam()

    def _init_model(self, params):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        self.cipher_operator = PaillierEncrypt()
        self.transfer_variable = HeteroFMTransferVariable()
        self.cipher.register_paillier_cipher(self.transfer_variable)
        self.converge_procedure.register_convergence(self.transfer_variable)
        self.batch_generator.register_batch_generator(self.transfer_variable)
        self.gradient_loss_operator.register_gradient_procedure(self.transfer_variable)

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

    def compute_fm(self, data_instances, model_weights):
        """ calculate w*x + (v*x)^2 - (v^2)*(x^2)
            data_instances: batch_size * feature_size
            model_weights: FM Model weights

            model_weights.w_'s shape is feature_size
            model_weights.embed_'s shape is feature_size * embed_size
        """

        def fm_func(features, embed):
            re = np.multiply(np.expand_dims(features, 1), embed)
            re = np.sum(re, 0)
            part1 = np.sum(np.power(re, 2))
            features_square = np.power(features, 2)
            embed_square = np.power(embed, 2)
            part2 = np.sum(np.dot(features_square, embed_square))
            return 0.5*(part1 - part2)

        wx = data_instances.mapValues(lambda v: np.dot(v.features, model_weights.w_) + model_weights.intercept_)
        fm = data_instances.mapValues(lambda v: fm_func(v.features, model_weights.embed_))

        return wx.join(fm, lambda wx_, fm_: wx_ + fm_)

    def compute_vx(self, data_instances, embed_):
        return data_instances.mapValues(lambda v: np.dot(v.features, embed_))



