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
from federatedml.statistic.data_overview import with_weight, scale_sample_weight
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal


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

    @staticmethod
    def load_data(data_instance):
        """
        return data_instance as original
        Parameters
        ----------
        data_instance: Table of Instance, input data
        """
        return data_instance

    def fit(self, data_instances, validate_data=None):
        """
        Train linR model of role guest
        Parameters
        ----------
        data_instances: Table of Instance, input data
        """

        LOGGER.info("Enter hetero_linR_guest fit")
        # self._abnormal_detection(data_instances)
        # self.header = self.get_header(data_instances)
        self.prepare_fit(data_instances, validate_data)
        self.callback_list.on_train_begin(data_instances, validate_data)

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        use_async = False
        if with_weight(data_instances):
            if self.model_param.early_stop == "diff":
                LOGGER.warning("input data with weight, please use 'weight_diff' for 'early_stop'.")
            data_instances = scale_sample_weight(data_instances)
            self.gradient_loss_operator.set_use_sample_weight()
            LOGGER.debug(f"instance weight scaled; use weighted gradient loss operator")
            # LOGGER.debug(f"data_instances after scale: {[v[1].weight for v in list(data_instances.collect())]}")
        elif len(self.component_properties.host_party_idlist) == 1:
            LOGGER.debug(f"set_use_async")
            self.gradient_loss_operator.set_use_async()
            use_async = True
        self.transfer_variable.use_async.remote(use_async)

        LOGGER.info("Generate mini-batch from input data")
        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)
        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_nums)

        LOGGER.info("Start initialize model.")
        LOGGER.info("fit_intercept:{}".format(self.init_param_obj.fit_intercept))
        model_shape = self.get_features_shape(data_instances)
        if not self.component_properties.is_warm_start:
            w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
            self.model_weights = LinearModelWeights(w, fit_intercept=self.fit_intercept, raise_overflow_error=False)
        else:
            self.callback_warm_start_init_iter(self.n_iter_)

        while self.n_iter_ < self.max_iter:
            self.callback_list.on_epoch_begin(self.n_iter_)
            LOGGER.info("iter:{}".format(self.n_iter_))
            # each iter will get the same batch_data_generator
            batch_data_generator = self.batch_generator.generate_batch_data()
            self.optimizer.set_iters(self.n_iter_)
            batch_index = 0
            for batch_data in batch_data_generator:
                # Start gradient procedure
                optim_guest_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_data,
                    self.cipher_operator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index
                )

                loss_norm = self.optimizer.loss_norm(self.model_weights)
                self.gradient_loss_operator.compute_loss(batch_data, self.n_iter_, batch_index, loss_norm)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_guest_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {}, is_converged: {}".format(self.n_iter_, self.is_converged))

            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1
            if self.stop_training:
                break

            if self.is_converged:
                break
        self.callback_list.on_train_end()

        self.set_summary(self.get_model_summary())

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Prediction of linR
        Parameters
        ----------
        data_instances: Table of Instance, input data
        predict_param: PredictParam, the setting of prediction.

        Returns
        ----------
        Table
            include input data label, predict results
        """
        LOGGER.info("Start predict ...")

        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        pred = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        host_preds = self.transfer_variable.host_partial_prediction.get(idx=-1)
        LOGGER.info("Get prediction from Host")

        for host_pred in host_preds:
            pred = pred.join(host_pred, lambda g, h: g + h)

        # predict_result = data_instances.join(pred, lambda d, pred: [d.label, pred, pred, {"label": pred}])
        predict_result = self.predict_score_to_output(data_instances=data_instances, predict_score=pred,
                                                      classes=None)
        return predict_result
