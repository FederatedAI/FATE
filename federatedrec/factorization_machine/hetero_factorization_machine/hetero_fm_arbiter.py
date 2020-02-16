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
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedrec.optim.gradient import hetero_fm_gradient_and_loss
from federatedrec.param.factorization_machine_param import HeteroFactorizationParam
from federatedrec.factorization_machine.hetero_factorization_machine.hetero_fm_base import HeteroFMBase


LOGGER = log_utils.getLogger()


class HeteroFMArbiter(HeteroFMBase):
    def __init__(self):
        super(HeteroFMArbiter, self).__init__()
        self.gradient_loss_operator = hetero_fm_gradient_and_loss.Arbiter()
        self.model_param = HeteroFactorizationParam()
        self.n_iter_ = 0
        self.header = None
        self.is_converged = False
        self.model_param_name = 'HeteroFactorizationMachineParam'
        self.model_meta_name = 'HeteroFactorizationMachineMeta'
        self.need_one_vs_rest = None
        self.in_one_vs_rest = False
        self.mode = consts.HETERO
        self.role = consts.ARBITER

        # attribute
        self.pre_loss = None
        self.cipher = paillier_cipher.Arbiter()
        self.batch_generator = batch_generator.Arbiter()
        self.converge_procedure = convergence.Arbiter()

    def fit(self, data_instances=None, validate_data=None):
        LOGGER.debug("Need one_vs_rest: {}".format(self.need_one_vs_rest))
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)
        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.in_one_vs_rest = True
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def fit_binary(self, data_instances=None, validate_data=None):
        """
        Train FM model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero fm model arbiter fit")

        self.cipher_operator = self.cipher.paillier_keygen(self.model_param.encrypt_param.key_length)
        self.batch_generator.initialize_batch_generator()
        validation_strategy = self.init_validation_strategy()

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
                # training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                # self.perform_subtasks(**training_info)

                loss_list = self.gradient_loss_operator.compute_loss(self.cipher_operator, self.n_iter_, batch_index)

                if len(loss_list) == 1:
                    if iter_loss is None:
                        iter_loss = loss_list[0]
                    else:
                        iter_loss += loss_list[0]

            # if converge
            if iter_loss is not None:
                iter_loss /= self.batch_generator.batch_num
                if not self.in_one_vs_rest:
                    self.callback_loss(self.n_iter_, iter_loss)

            if self.model_param.early_stop == 'weight_diff':
                weight_diff = fate_operator.norm(total_gradient)
                LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                                                                                weight_diff, self.is_converged))
                if weight_diff < self.model_param.tol:
                    self.is_converged = True
            else:
                self.is_converged = self.converge_func.is_converge(iter_loss)
                LOGGER.info("iter: {},  loss:{}, is_converged: {}".format(self.n_iter_, iter_loss, self.is_converged))

            self.converge_procedure.sync_converge_info(self.is_converged, suffix=(self.n_iter_,))

            validation_strategy.validate(self, self.n_iter_)

            self.n_iter_ += 1
            if self.is_converged:
                break


