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
from federatedml.linear_model.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim.gradient import hetero_linr_gradient_and_loss
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
        self.gradient_loss_operator = hetero_linr_gradient_and_loss.Guest()
        self.converge_procedure = convergence.Guest()
        self.encrypted_calculator = None

    @staticmethod
    def load_data(data_instance):
        """
        return data_instance as original
        Parameters
        ----------
        data_instance: DTable of Instance, input data
        """
        return data_instance

    def fit(self, data_instances, validate_data=None):
        """
        Train linR model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_linR_guest fit")
        self._abnormal_detection(data_instances)
        self.header = self.get_header(data_instances)

        self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        LOGGER.info("Generate mini-batch from input data")
        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)
        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_nums)

        self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_generator.batch_nums)]

        LOGGER.info("Start initialize model.")
        LOGGER.info("fit_intercept:{}".format(self.init_param_obj.fit_intercept))
        model_shape = self.get_features_shape(data_instances)
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        self.model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            # each iter will get the same batch_data_generator
            batch_data_generator = self.batch_generator.generate_batch_data()
            self.optimizer.set_iters(self.n_iter_)
            batch_index = 0
            for batch_data in batch_data_generator:
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)

                # Start gradient procedure
                optim_guest_gradient, _, _ = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_feat_inst,
                    self.encrypted_calculator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index
                )

                loss_norm = self.optimizer.loss_norm(self.model_weights)
                self.gradient_loss_operator.compute_loss(data_instances, self.n_iter_, batch_index, loss_norm)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_guest_gradient)
                batch_index += 1
                LOGGER.debug(
                    "model_weights, iters: {}, update_model: {}".format(self.n_iter_, self.model_weights.unboxed))

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))

            LOGGER.debug("model weights is {}".format(self.model_weights.coef_))

            if self.validation_strategy:
                LOGGER.debug('LinR guest running validation')
                self.validation_strategy.validate(self, self.n_iter_)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            self.n_iter_ += 1
            if self.is_converged:
                break
        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)

    def predict(self, data_instances):
        """
        Prediction of linR
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        predict_param: PredictParam, the setting of prediction.

        Returns
        ----------
        DTable
            include input data label, predict results
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)
        pred = self.compute_wx(data_features, self.model_weights.coef_, self.model_weights.intercept_)
        host_preds = self.transfer_variable.host_partial_prediction.get(idx=-1)
        LOGGER.info("Get prediction from Host")

        for host_pred in host_preds:
            pred = pred.join(host_pred, lambda g, h: g + h)

        predict_result = data_instances.join(pred, lambda d, pred: [d.label, pred, pred, {"label": pred}])
        return predict_result
