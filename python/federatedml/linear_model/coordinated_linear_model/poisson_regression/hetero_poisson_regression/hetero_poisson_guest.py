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

import copy
import numpy as np

from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.coordinated_linear_model.poisson_regression. \
    hetero_poisson_regression.hetero_poisson_base import HeteroPoissonBase
from federatedml.optim.gradient import hetero_poisson_gradient_and_loss
from federatedml.statistic.data_overview import with_weight
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal


class HeteroPoissonGuest(HeteroPoissonBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        self.role = consts.GUEST
        self.cipher = paillier_cipher.Guest()
        self.batch_generator = batch_generator.Guest()
        self.gradient_loss_operator = hetero_poisson_gradient_and_loss.Guest()
        self.converge_procedure = convergence.Guest()

    def fit(self, data_instances, validate_data=None):
        """
        Train poisson model of role guest
        Parameters
        ----------
        data_instances: Table of Instance, input data
        """

        LOGGER.info("Enter hetero_poisson_guest fit")
        # self._abnormal_detection(data_instances)
        # self.header = copy.deepcopy(self.get_header(data_instances))
        self.prepare_fit(data_instances, validate_data)
        self.callback_list.on_train_begin(data_instances, validate_data)

        if with_weight(data_instances):
            LOGGER.warning("input data with weight. Poisson regression does not support weighted training.")

        self.exposure_index = self.get_exposure_index(self.header, self.exposure_colname)
        exposure_index = self.exposure_index
        if exposure_index > -1:
            self.header.pop(exposure_index)
            LOGGER.info("Guest provides exposure value.")
        exposure = data_instances.mapValues(lambda v: HeteroPoissonBase.load_exposure(v, exposure_index))
        data_instances = data_instances.mapValues(lambda v: HeteroPoissonBase.load_instance(v, exposure_index))

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        LOGGER.info("Generate mini-batch from input data")
        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)

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
                # compute offset of this batch
                batch_offset = exposure.join(batch_data, lambda ei, d: HeteroPoissonBase.safe_log(ei))

                # Start gradient procedure
                optimized_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_data,
                    self.cipher_operator,
                    self.model_weights,
                    self.optimizer,
                    self.n_iter_,
                    batch_index,
                    batch_offset
                )
                # LOGGER.debug("iteration:{} Guest's gradient: {}".format(self.n_iter_, optimized_gradient))
                loss_norm = self.optimizer.loss_norm(self.model_weights)
                self.gradient_loss_operator.compute_loss(batch_data, self.model_weights, self.n_iter_,
                                                         batch_index, batch_offset, loss_norm)

                self.model_weights = self.optimizer.update_model(self.model_weights, optimized_gradient)

                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))

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
        Prediction of Poisson
        Parameters
        ----------
        data_instances: Table of Instance, input data

        Returns
        ----------
        Table
            include input data label, predict results
        """
        LOGGER.info("Start predict ...")

        self._abnormal_detection(data_instances)
        header = data_instances.schema.get("header")
        self.exposure_index = self.get_exposure_index(header, self.exposure_colname)
        exposure_index = self.exposure_index

        exposure = data_instances.mapValues(lambda v: HeteroPoissonBase.load_exposure(v, exposure_index))

        data_instances = self.align_data_header(data_instances, self.header)
        data_instances = data_instances.mapValues(lambda v: HeteroPoissonBase.load_instance(v, exposure_index))

        # pred_guest = self.compute_mu(data_instances, self.model_weights.coef_, self.model_weights.intercept_, exposure)
        # pred_host = self.transfer_variable.host_partial_prediction.get(idx=0)

        wx_guest = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        wx_host = self.transfer_variable.host_partial_prediction.get(idx=0)
        LOGGER.info("Get prediction from Host")

        # pred = pred_guest.join(pred_host, lambda g, h: g * h)
        pred = wx_guest.join(wx_host, lambda g, h: np.exp(g + h))
        pred = pred.join(exposure, lambda mu, b: mu * b)
        # predict_result = data_instances.join(pred, lambda d, p: [d.label, p, p, {"label": p}])
        predict_result = self.predict_score_to_output(data_instances=data_instances, predict_score=pred,
                                                      classes=None)
        return predict_result
