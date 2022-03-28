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
from federatedml.linear_model.coordinated_linear_model.linear_regression.hetero_linear_regression.hetero_linr_base import \
    HeteroLinRBase
from federatedml.optim.gradient import hetero_linr_gradient_and_loss
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroLinRHost(HeteroLinRBase):
    def __init__(self):
        super(HeteroLinRHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []
        self.role = consts.HOST

        self.cipher = paillier_cipher.Host()
        self.batch_generator = batch_generator.Host()
        self.gradient_loss_operator = hetero_linr_gradient_and_loss.Host()
        self.converge_procedure = convergence.Host()

    def fit(self, data_instances, validate_data=None):
        """
        Train linear regression model of role host
        Parameters
        ----------
        data_instances: Table of Instance, input data
        """

        LOGGER.info("Enter hetero_linR host")
        # self._abnormal_detection(data_instances)
        # self.header = self.get_header(data_instances)
        self.prepare_fit(data_instances, validate_data)
        self.callback_list.on_train_begin(data_instances, validate_data)

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        if self.transfer_variable.use_async.get(idx=0):
            LOGGER.debug(f"set_use_async")
            self.gradient_loss_operator.set_use_async()

        self.batch_generator.initialize_batch_generator(data_instances)
        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_nums)

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
            self.callback_list.on_epoch_begin(self.n_iter_)
            LOGGER.info("iter:" + str(self.n_iter_))
            self.optimizer.set_iters(self.n_iter_)
            batch_data_generator = self.batch_generator.generate_batch_data()
            batch_index = 0
            for batch_data in batch_data_generator:
                optim_host_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_data,
                    self.cipher_operator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index)

                self.gradient_loss_operator.compute_loss(self.model_weights, self.optimizer, self.n_iter_, batch_index,
                                                         self.cipher_operator)

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
        # LOGGER.debug(f"summary content is: {self.summary()}")

    def predict(self, data_instances):
        """
        Prediction of linR
        Parameters
        ----------
        data_instances:Table of Instance, input data
        """
        self.transfer_variable.host_partial_prediction.disable_auto_clean()

        LOGGER.info("Start predict ...")

        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)

        pred_host = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        self.transfer_variable.host_partial_prediction.remote(pred_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote partial prediction to Guest")
