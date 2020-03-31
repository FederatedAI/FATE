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
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.validation_strategy import ValidationStrategy

LOGGER = log_utils.getLogger()


class HeteroBaseArbiter(BaseLinearModel):
    def __init__(self):
        super(BaseLinearModel, self).__init__()
        self.role = consts.ARBITER

        # attribute
        self.pre_loss = None
        self.loss_history = []
        self.cipher = paillier_cipher.Arbiter()
        self.batch_generator = batch_generator.Arbiter()
        self.gradient_loss_operator = None
        self.converge_procedure = convergence.Arbiter()

    def perform_subtasks(self, **training_info):
        """
        performs any tasks that the arbiter is responsible for.

        This 'perform_subtasks' function serves as a handler on conducting any task that the arbiter is responsible
        for. For example, for the 'perform_subtasks' function of 'HeteroDNNLRArbiter' class located in
        'hetero_dnn_lr_arbiter.py', it performs some works related to updating/training local neural networks of guest
        or host.

        For this particular class, the 'perform_subtasks' function will do nothing. In other words, no subtask is
        performed by this arbiter.

        :param training_info: a dictionary holding training information
        """
        pass

    # def run(self, component_parameters=None, args=None):
    #     self._init_runtime_parameters(component_parameters)
    #
    #     if self.need_cv:
    #         LOGGER.info("Task is cross validation.")
    #         self.cross_validation(None)
    #         return
    #
    #     elif not "model" in args:
    #         LOGGER.info("Task is fit")
    #         self.set_flowid('fit')
    #         self.fit()
    #     else:
    #         LOGGER.info("Task is predict, No need for arbiter to involve.")

    def init_validation_strategy(self, train_data=None, validate_data=None):
        validation_strategy = ValidationStrategy(self.role, self.mode, self.validation_freqs, self.early_stopping_rounds)
        return validation_strategy

    def fit(self, data_instances=None, validate_data=None):
        """
        Train linear model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero linear model arbiter fit")

        self.cipher_operator = self.cipher.paillier_keygen(self.model_param.encrypt_param.key_length)
        self.batch_generator.initialize_batch_generator()
        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_num)

        self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        while self.n_iter_ < self.max_iter:
            iter_loss = None
            batch_data_generator = self.batch_generator.generate_batch_data()
            total_gradient = None
            self.optimizer.set_iters(self.n_iter_)
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
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.perform_subtasks(**training_info)

                loss_list = self.gradient_loss_operator.compute_loss(self.cipher_operator, self.n_iter_, batch_index)

                if len(loss_list) == 1:
                    if iter_loss is None:
                        iter_loss = loss_list[0]
                    else:
                        iter_loss += loss_list[0]
                        # LOGGER.info("Get loss from guest:{}".format(de_loss))

            # if converge
            if iter_loss is not None:
                iter_loss /= self.batch_generator.batch_num
                if self.need_call_back_loss:
                    self.callback_loss(self.n_iter_, iter_loss)
                self.loss_history.append(iter_loss)

            if self.model_param.early_stop == 'weight_diff':
                LOGGER.debug("total_gradient: {}".format(total_gradient))
                weight_diff = fate_operator.norm(total_gradient)
                LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                                                                                weight_diff, self.is_converged))
                if weight_diff < self.model_param.tol:
                    self.is_converged = True
            else:
                if iter_loss is None:
                    raise ValueError("Multiple host situation, loss early stop function is not available."
                                     "You should use 'weight_diff' instead")
                self.is_converged = self.converge_func.is_converge(iter_loss)
                LOGGER.info("iter: {},  loss:{}, is_converged: {}".format(self.n_iter_, iter_loss, self.is_converged))

            self.converge_procedure.sync_converge_info(self.is_converged, suffix=(self.n_iter_,))

            if self.validation_strategy:
                LOGGER.debug('Linear Arbiter running validation')
                self.validation_strategy.validate(self, self.n_iter_)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            self.n_iter_ += 1
            if self.is_converged:
                break
        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)
        LOGGER.debug("finish running linear model arbiter")
