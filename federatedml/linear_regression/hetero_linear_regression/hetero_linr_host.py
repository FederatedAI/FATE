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

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.framework.hetero.procedure import loss_computer_linR, convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim.gradient import hetero_gradient_procedure
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.util import consts

import time

LOGGER = log_utils.getLogger()


class HeteroLinRHost(HeteroLinRBase):
    def __init__(self):
        super(HeteroLinRHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []
        self.wx = None
        self.batch_index_list = []
        self.role = consts.HOST

        self.cipher = paillier_cipher.Host()
        self.batch_generator = batch_generator.Host()
        self.gradient_procedure = hetero_gradient_procedure.Host()
        self.loss_computer = loss_computer.Host()
        self.converge_procedure = convergence.Host()

    def fit(self, data_instances):
        """
        Train linear regression model of role host
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_linr host")
        self._abnormal_detection(data_instances)

        self.header = self.get_header(data_instances)
        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)

        self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_generator.batch_nums)]

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)
        # host does not hold intercept
        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False
        self.lr_variables = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))
            batch_data_generator = self.batch_generator.generate_batch_data()
            batch_index = 0

            for batch_data in batch_data_generator:
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)
                optim_host_gradient, loss = self.gradient_procedure.compute_gradient_procedure(
                    batch_feat_inst, self.lr_variables, self.compute_wx,
                    self.encrypted_calculator, self.n_iter_, batch_index)

                self.loss_computer.sync_loss_info(self.lr_variables, loss, self.n_iter_, batch_index,
                                                  self.cipher, self.optimizer)

                self.lr_variables = self.optimizer.update_model(self.lr_variables, optim_host_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))

            LOGGER.info("Get is_converged flag from arbiter:{}".format(self.is_converged))

            self.n_iter_ += 1
            LOGGER.info("iter: {}, is_converged: {}".format(self.n_iter_, self.is_converged))
            if self.is_converged:
                break
        #LOGGER.info("host model coef: {}".format(self.coef_))
        LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))

    def predict(self, data_instances):
        """
        Prediction of linear regression
        Parameters
        ----------
        data_instances:DTable of Instance, input data
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)

        partial_prediction = self.compute_wx(data_features, self.coef_, self.intercept_)
        self.transfer_variable.host_partial_prediction.remote(partial_prediction, role=consts.GUEST, idx=0)
        LOGGER.info("Remote partial_prediction to Guest")
