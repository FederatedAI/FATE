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

from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.coordinated_linear_model.poisson_regression. \
    hetero_poisson_regression.hetero_poisson_base import HeteroPoissonBase
from federatedml.optim.gradient import hetero_poisson_gradient_and_loss
from federatedml.util import LOGGER
from federatedml.util import consts


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

    def fit(self, data_instances, validate_data=None):
        """
        Train poisson regression model of role host
        Parameters
        ----------
        data_instances: Table of Instance, input data
        """

        LOGGER.info("Enter hetero_poisson host")
        # self._abnormal_detection(data_instances)
        # self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        # self.header = self.get_header(data_instances)
        self.prepare_fit(data_instances, validate_data)
        self.callback_list.on_train_begin(data_instances, validate_data)

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        self.batch_generator.initialize_batch_generator(data_instances)

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)
        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False
        if not self.component_properties.is_warm_start:
            w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
            self.model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept, raise_overflow_error=False)
        else:
            self.callback_warm_start_init_iter(self.n_iter_)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))

            batch_data_generator = self.batch_generator.generate_batch_data()
            self.optimizer.set_iters(self.n_iter_)

            batch_index = 0
            for batch_data in batch_data_generator:
                self.callback_list.on_epoch_begin(self.n_iter_)
                LOGGER.info("iter:" + str(self.n_iter_))
                optim_host_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_data,
                    self.cipher_operator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index)

                self.gradient_loss_operator.compute_loss(batch_data, self.model_weights,
                                                         self.optimizer,
                                                         self.n_iter_, batch_index, self.cipher_operator)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_host_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))

            LOGGER.info("Get is_converged flag from arbiter:{}".format(self.is_converged))

            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1
            if self.stop_training:
                break

            LOGGER.info("iter: {}, is_converged: {}".format(self.n_iter_, self.is_converged))
            if self.is_converged:
                break
        self.callback_list.on_train_end()
        self.set_summary(self.get_model_summary())

    def predict(self, data_instances):
        """
        Prediction of poisson
        Parameters
        ----------
        data_instances:Table of Instance, input data
        """
        self.transfer_variable.host_partial_prediction.disable_auto_clean()
        LOGGER.info("Start predict ...")

        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        # pred_host = self.compute_mu(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        pred_host = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        self.transfer_variable.host_partial_prediction.remote(pred_host, role=consts.GUEST, idx=0)

        LOGGER.info("Remote partial prediction to Guest")
