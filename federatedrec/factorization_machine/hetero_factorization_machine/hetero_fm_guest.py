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

from arch.api.utils import log_utils
from federatedml.util import consts
from federatedml.optim import activation
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedrec.optim.gradient import hetero_fm_gradient_and_loss
from federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights
from federatedrec.factorization_machine.hetero_factorization_machine.hetero_fm_base import HeteroFMBase


LOGGER = log_utils.getLogger()


class HeteroFMGuest(HeteroFMBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        self.role = consts.GUEST
        self.cipher = paillier_cipher.Guest()
        self.batch_generator = batch_generator.Guest()
        self.gradient_loss_operator = hetero_fm_gradient_and_loss.Guest()
        self.converge_procedure = convergence.Guest()
        self.encrypted_calculator = None

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

    def fit(self, data_instances, validate_data=None):
        """
        Train fm model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_fm_guest fit")
        self._abnormal_detection(data_instances)
        self.header = self.get_header(data_instances)

        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.in_one_vs_rest = True
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data=None):
        LOGGER.info("Enter hetero_fm_guest fit")
        self.header = self.get_header(data_instances)

        validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        data_instances = data_instances.mapValues(HeteroFMGuest.load_data)
        LOGGER.debug(f"MODEL_STEP After load data, data count: {data_instances.count()}")
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

        # intercept is initialized within FactorizationMachineWeights.
        # Skip initializer's intercept part.
        fit_intercept = False
        if self.init_param_obj.fit_intercept:
            fit_intercept = True
            self.init_param_obj.fit_intercept = False

        w_ = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        embed_ = np.random.normal(scale=1 / np.sqrt(self.init_param_obj.embed_size),
                                  size=(model_shape, self.init_param_obj.embed_size))
        self.model_weights = \
            FactorizationMachineWeights(w_, embed_, fit_intercept=fit_intercept)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            batch_data_generator = self.batch_generator.generate_batch_data()
            self.optimizer.set_iters(self.n_iter_)
            batch_index = 0
            for batch_data in batch_data_generator:
                LOGGER.debug(f"MODEL_STEP In Batch {batch_index}, batch data count: {batch_data.count()}")
                # Start gradient procedure
                LOGGER.debug("iter: {}, before compute gradient, data count: {}".format(self.n_iter_,
                                                                                        batch_data.count()))
                # optim_guest_gradient, fore_gradient, host_forwards = self.gradient_loss_operator. \
                optim_guest_gradient, fore_gradient = self.gradient_loss_operator. \
                    compute_gradient_procedure(
                        batch_data,
                        self.encrypted_calculator,
                        self.model_weights,
                        self.optimizer,
                        self.n_iter_,
                        batch_index
                )
                LOGGER.debug('optim_guest_gradient: {}'.format(optim_guest_gradient))

                loss_norm = self.optimizer.loss_norm(self.model_weights)
                self.gradient_loss_operator.compute_loss(data_instances, self.n_iter_, batch_index, loss_norm)

                # clip gradient
                if self.model_param.clip_gradient and self.model_param.clip_gradient > 0:
                    optim_guest_gradient = np.maximum(optim_guest_gradient, -self.model_param.clip_gradient)
                    optim_guest_gradient = np.minimum(optim_guest_gradient, self.model_param.clip_gradient)

                _model_weights = self.optimizer.update_model(self.model_weights, optim_guest_gradient)
                self.model_weights.update(_model_weights)
                batch_index += 1
                LOGGER.debug("fm_weight, iters: {}, update_model: {}".format(self.n_iter_, self.model_weights.unboxed))

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))

            validation_strategy.validate(self, self.n_iter_)

            self.n_iter_ += 1
            if self.is_converged:
                break

        LOGGER.debug("Final fm weights: {}".format(self.model_weights.unboxed))

    def predict(self, data_instances):
        """
        Prediction of fm
        Parameters
        ----------
        data_instances:DTable of Instance, input data

        Returns
        ----------
        DTable
            include input data label, predict probably, label
        """
        LOGGER.info("Start predict is a one_vs_rest task: {}".format(self.need_one_vs_rest))
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result

        prob_guest = self.compute_fm(data_instances, self.model_weights)
        vx_guest = self.compute_vx(data_instances, self.model_weights.embed_)
        prob_guest = prob_guest.join(vx_guest, lambda a, b: (a, b))

        host_probs = self.transfer_variable.host_prob.get(idx=-1)

        LOGGER.info("Get probability from Host")

        # guest probability
        assert(len(host_probs)==1, "Currently Hetero FM only support single host.")
        host_prob = host_probs[0]

        pred_prob = prob_guest.join(host_prob, lambda g, h: g[0] + h[0] + np.dot(h[1], g[1]))
        pred_prob = pred_prob.mapValues(lambda p: activation.sigmoid(p))
        pred_label = pred_prob.mapValues(lambda x: 1 if x > self.model_param.predict_param.threshold else 0)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"0": (1 - x[1]), "1": x[1]}])

        return predict_result
