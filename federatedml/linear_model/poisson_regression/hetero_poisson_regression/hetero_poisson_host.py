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
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.poisson_regression.hetero_poisson_regression.hetero_poisson_base import HeteroPoissonBase
from federatedml.optim.gradient import hetero_poisson_gradient_and_loss
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroPoissonHost(HeteroPoissonBase):
    def __init__(self):
        super(HeteroPoissonHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []
        self.role = consts.HOST

        self.cipher = paillier_cipher.Host()
        self.batch_generator = batch_generator.Host()
        self.gradient_loss_operator = hetero_poisson_gradient_and_loss.Host()
        self.converge_procedure = convergence.Host()
        self.encrypted_calculator = None

    def fit(self, data_instances, validate_data=None):
        """
        Train poisson regression model of role host
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_poisson host")
        self._abnormal_detection(data_instances)

        self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        self.header = self.get_header(data_instances)
        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        self.batch_generator.initialize_batch_generator(data_instances)

        self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_generator.batch_nums)]

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)
        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        self.model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))

            batch_data_generator = self.batch_generator.generate_batch_data()
            self.optimizer.set_iters(self.n_iter_)

            batch_index = 0
            for batch_data in batch_data_generator:
                batch_feat_inst = self.transform(batch_data)
                optim_host_gradient, _ = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_feat_inst,
                    self.encrypted_calculator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index)

                self.gradient_loss_operator.compute_loss(batch_feat_inst, self.model_weights,
                                                         self.encrypted_calculator, self.optimizer,
                                                         self.n_iter_, batch_index, self.cipher_operator)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_host_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))

            LOGGER.info("Get is_converged flag from arbiter:{}".format(self.is_converged))

            if self.validation_strategy:
                LOGGER.debug('Poisson host running validation')
                self.validation_strategy.validate(self, self.n_iter_)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            self.n_iter_ += 1
            LOGGER.info("iter: {}, is_converged: {}".format(self.n_iter_, self.is_converged))
            if self.is_converged:
                break

        if not self.is_converged:
            LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))

        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)

    def predict(self, data_instances):
        """
        Prediction of poisson
        Parameters
        ----------
        data_instances:DTable of Instance, input data
        """
        self.transfer_variable.host_partial_prediction.disable_auto_clean()
        LOGGER.info("Start predict ...")
        data_features = self.transform(data_instances)
        pred_host = self.compute_mu(data_features, self.model_weights.coef_, self.model_weights.intercept_)
        self.transfer_variable.host_partial_prediction.remote(pred_host, role=consts.GUEST, idx=0)

        LOGGER.info("Remote partial prediction to Guest")
