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
from federatedml.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim import activation
from federatedml.optim.gradient import hetero_gradient_procedure
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRGuest(HeteroLinRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        # self.guest_forward = None
        self.role = consts.GUEST
        self.cipher = paillier_cipher.Guest()
        self.batch_generator = batch_generator.Guest()
        self.gradient_procedure = hetero_gradient_procedure.Guest()
        self.loss_computer = loss_computer.Guest()
        self.converge_procedure = convergence.Guest()

    @staticmethod
    def load_data(data_instance):
        """
        set the negative label to -1
        Parameters
        ----------
        data_instance: DTable of Instance, input data
        """
        if data_instance.label != 1:
            data_instance.label = -1
        return data_instance

    def fit(self, data_instances):
        """
        Train linR model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_linR_guest fit")
        self._abnormal_detection(data_instances)
        self.header = self.get_header(data_instances)
        data_instances = data_instances.mapValues(HeteroLinRGuest.load_data)

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        LOGGER.info("Generate mini-batch from input data")
        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)
        self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_generator.batch_nums)]

        LOGGER.info("Start initialize model.")
        LOGGER.info("fit_intercept:{}".format(self.init_param_obj.fit_intercept))
        model_shape = self.get_features_shape(data_instances)
        self.linR_variables = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            # each iter will get the same batach_data_generator
            batch_data_generator = self.batch_generator.generate_batch_data()

            batch_index = 0
            for batch_data in batch_data_generator:
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)

                # Start gradient procedure
                optim_guest_gradient, loss, fore_gradient = self.gradient_procedure.compute_gradient_procedure(
                    batch_feat_inst,
                    self.linR_variables,
                    self.compute_wx,
                    self.encrypted_calculator,
                    self.n_iter_,
                    batch_index
                )

                self.loss_computer.sync_loss_info(self.linR_variables, loss, self.n_iter_, batch_index, self.optimizer)

                self.linR_variables = self.optimizer.update_model(self.linR_variables, optim_guest_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
            self.n_iter_ += 1
            if self.is_converged:
                break

    def predict(self, data_instances):
        """
        Prediction of linR
        Parameters
        ----------
        data_instances:DTable of Instance, input data
        predict_param: PredictParam, the setting of prediction.

        Returns
        ----------
        DTable
            include input data label, predict probably, label
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)
        pred_guest = self.compute_wx(data_features, self.linR_variables.coef_, self.linR_variables.intercept_)
        pred_host = self.transfer_variable.host_partial_prediction.get(idx=0)

        LOGGER.info("Get prediction from Host")

        pred = pred_guest.join(pred_host, lambda g, h: g + h)
        predict_result = data_instances.join(pred, lambda x, y: [x.label, y])

        return predict_result
