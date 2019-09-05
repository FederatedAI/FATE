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
from federatedml.framework.hetero.procedure import loss_computer, convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.logistic_regression.hetero_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim.gradient import hetero_gradient_procedure
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLRArbiter(HeteroLRBase):
    def __init__(self):
        # LogisticParamChecker.check_param(logistic_params)
        super(HeteroLRArbiter, self).__init__()
        self.role = consts.ARBITER

        # attribute
        self.pre_loss = None

        self.cipher = paillier_cipher.Arbiter()
        self.batch_generator = batch_generator.Arbiter()
        self.gradient_procedure = hetero_gradient_procedure.Arbiter()
        self.loss_computer = loss_computer.Arbiter()
        self.converge_procedure = convergence.Arbiter()

    def perform_subtasks(self, **training_info):
        """
        performs any tasks that the arbiter is responsible for.

        This 'perform_subtasks' function serves as a handler on conducting any task that the arbiter is responsible
        for. For example, for the 'perform_subtasks' function of 'HeteroDNNLRArbiter' class located in
        'hetero_dnn_lr_arbiter.py', it performs some works related to updating/training local neural networks of guest
        or host.

        For this particular class (i.e., 'HeteroLRArbiter') that serves as a base arbiter class for neural-networks-based
        hetero-logistic-regression model, the 'perform_subtasks' function will do nothing. In other words, no subtask is
        performed by this arbiter.

        :param training_info: a dictionary holding training information
        """
        pass

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)

        if self.need_cv:
            LOGGER.info("Task is cross validation.")
            self.cross_validation(None)
            return

        if self.need_one_vs_rest:
            LOGGER.info("Task is one_vs_rest fit")
            if not "model" in args:
                self.one_vs_rest_fit()
        elif not "model" in args:
            LOGGER.info("Task is fit")
            self.set_flowid('train')
            self.fit()
        else:
            LOGGER.info("Task is transform")

    def fit(self, data_instances=None):
        """
        Train lr model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_lr_arbiter fit")

        self.cipher_operator = self.cipher.paillier_keygen(self.model_param.encrypt_param.key_length)
        self.batch_generator.initialize_batch_generator()

        while self.n_iter_ < self.max_iter:
            iter_loss = 0
            batch_data_generator = self.batch_generator.generate_batch_data()

            for batch_index in batch_data_generator:
                # Compute and Transfer gradient info
                self.gradient_procedure.compute_gradient_procedure(self.cipher_operator,
                                                                   self.optimizer,
                                                                   self.n_iter_,
                                                                   batch_index)

                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.perform_subtasks(**training_info)

                loss = self.loss_computer.sync_loss_info(self.n_iter_, batch_index)

                de_loss = self.cipher_operator.decrypt(loss)
                iter_loss += de_loss
                # LOGGER.info("Get loss from guest:{}".format(de_loss))

            # if converge
            loss = iter_loss / self.batch_generator.batch_num

            self.callback_loss(self.n_iter_, loss)

            if self.converge_func.is_converge(loss):
                self.is_converged = True
            LOGGER.info("iter: {},  loss:{}, is_converged: {}".format(self.n_iter_, loss, self.is_converged))

            self.converge_procedure.sync_converge_info(self.is_converged, suffix=(self.n_iter_,))
            self.n_iter_ += 1
            if self.is_converged:
                break
