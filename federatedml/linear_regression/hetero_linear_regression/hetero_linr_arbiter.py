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
from federatedml.framework.hetero.procedure import loss_computer, convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim.gradient import hetero_gradient_procedure
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRArbiter(HeteroLinRBase):
    def __init__(self):
        super(HeteroLinRArbiter, self).__init__()
        self.role = consts.ARBITER

        # attribute
        self.pre_loss = None

        self.cipher = paillier_cipher.Arbiter()
        self.batch_generator = batch_generator.Arbiter()
        self.gradient_procedure = hetero_gradient_procedure.Arbiter()
        self.loss_computer = loss_computer.Arbiter()
        self.converge_procedure = convergence.Arbiter()

    def perform_subtasks(self, **training_info):
        pass

    def run(self, component_parameters=None, args=None):
        """
        check mode of task
        :param component_parameters: for cross validation
        :param args: string, task input
        :return:
        """
        self._init_runtime_parameters(component_parameters)

        if self.need_cv:
            LOGGER.info("Task is cross validation")
            self.cross_validation(None)
            return
        if "model" not in args:
            LOGGER.info("Task is fit")
            self.set_flowid('train')
            self.fit()
        else:
            LOGGER.info("Task is transform")

    def fit(self, data_instances=None):
        """
        Train linear regression model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """
        LOGGER.info("Enter hetero_linr_arbiter fit")

        self.cipher_operator = self.cipher.paillier_keygen(self.model_param.encrypt_param.key_length)
        self.batch_generator.initialize_batch_generator()

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            iter_loss = 0
            batch_data_generator = self.batch_generator.generate_batch_data()
            for batch_index in batch_data_generator:
                # Compute and Transfer gradient info
                self.gradient_procedure.compute_gradient_procedure(self.cipher_operator,
                                                                   self.optimizer,
                                                                   self.n_iter_,
                                                                   batch_index)
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}

                loss = self.loss_computer.sync_loss_info(self.n_iter_, batch_index)
                de_loss = self.cipher_operator.decrypt(loss)
                iter_loss += de_loss
                self.perform_subtasks(**training_info)

            # if converge
            loss = iter_loss / self.batch_generator.batch_num

            self.callback_loss(self.n_iter_, loss)

            if self.converge_func.is_converge(loss):
                self.is_converged = True
            LOGGER.info("iter:{}, loss:{}, is_converged:{}".format(self.n_iter_, loss, self.is_converged))

            self.converge_procedure.sync_converge_info(self.is_converged, suffix=(self.n_iter_,))
            self.n_iter_ += 1
            if self.is_converged:
                break
