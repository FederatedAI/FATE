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

from arch.api.utils import log_utils
from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_model.poisson_regression.hetero_poisson_regression.hetero_poisson_base import HeteroPoissonBase
from federatedml.optim.gradient import hetero_poisson_gradient_and_loss
from federatedml.util import consts
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HeteroPoissonArbiter(HeteroPoissonBase):
    def __init__(self):
        super(HeteroPoissonArbiter, self).__init__()
        self.role = consts.ARBITER

        # attribute
        self.pre_loss = None

        self.cipher = paillier_cipher.Arbiter()
        self.batch_generator = batch_generator.Arbiter()
        self.gradient_loss_operator = hetero_poisson_gradient_and_loss.Arbiter()
        self.converge_procedure = convergence.Arbiter()

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
            self.set_flowid('fit')
            self.fit()
        else:
            LOGGER.info("Task is transform")

    def fit(self, data_instances=None, validate_data=None):
        """
        Train poisson regression model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """
        LOGGER.info("Enter hetero_poisson_arbiter fit")

        self.cipher_operator = self.cipher.paillier_keygen(self.model_param.encrypt_param.key_length)
        self.batch_generator.initialize_batch_generator()

        validation_strategy = self.init_validation_strategy()
        
        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            iter_loss =None
            batch_data_generator = self.batch_generator.generate_batch_data()
            total_gradient = None
            self.optimizer.set_iters(self.n_iter_ + 1)
            for batch_index in batch_data_generator:
                # Compute and Transfer gradient info
                gradient = self.gradient_loss_operator.compute_gradient_procedure(self.cipher_operator,
                                                                   self.optimizer,
                                                                   self.n_iter_,
                                                                   batch_index)
                if total_gradient is None:
                    total_gradient = gradient
                else:
                    total_gradient = total_gradient + gradient

                loss_list = self.gradient_loss_operator.compute_loss(self.cipher_operator, self.n_iter_, batch_index)
                if iter_loss is None:
                    iter_loss = loss_list[0]
                else:
                    iter_loss = iter_loss + loss_list[0]

            # if converge
            if iter_loss is not None:
                iter_loss = iter_loss / self.batch_generator.batch_num
                self.callback_loss(self.n_iter_, iter_loss)

            if self.model_param.converge_func == 'weight_diff':
                weight_diff = fate_operator.norm(total_gradient)
                if weight_diff < self.model_param.eps:
                    self.is_converged = True
                LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                                                                                weight_diff, self.is_converged))

            else:
                self.is_converged = self.converge_func.is_converge(iter_loss)
                LOGGER.info("iter: {},  loss:{}, is_converged: {}".format(self.n_iter_, iter_loss, self.is_converged))

            self.converge_procedure.sync_converge_info(self.is_converged, suffix=(self.n_iter_,))
            validation_strategy.validate(self, self.n_iter_)
            
            self.n_iter_ += 1
            if self.is_converged:
                break
