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
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedrec.optim.gradient import hetero_fm_gradient_and_loss
from federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights
from federatedrec.factorization_machine.hetero_factorization_machine.hetero_fm_base import HeteroFMBase


LOGGER = log_utils.getLogger()


class HeteroFMHost(HeteroFMBase):
    def __init__(self):
        super(HeteroFMHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []
        self.role = consts.HOST

        self.cipher = paillier_cipher.Host()
        self.batch_generator = batch_generator.Host()
        self.gradient_loss_operator = hetero_fm_gradient_and_loss.Host()
        self.converge_procedure = convergence.Host()
        self.encrypted_calculator = None

    def fit(self, data_instances, validate_data=None):
        """
        Train fm model of role host
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_fm host")
        self.header = self.get_header(data_instances)

        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.in_one_vs_rest = True
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data):
        self._abnormal_detection(data_instances)

        validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        LOGGER.debug(f"MODEL_STEP Start fin_binary, data count: {data_instances.count()}")

        self.header = self.get_header(data_instances)
        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        self.batch_generator.initialize_batch_generator(data_instances)

        self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_generator.batch_nums)]

        LOGGER.info("Start initialize model.")
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
            LOGGER.info("iter:" + str(self.n_iter_))
            batch_data_generator = self.batch_generator.generate_batch_data()
            batch_index = 0
            self.optimizer.set_iters(self.n_iter_)
            for batch_data in batch_data_generator:
                LOGGER.debug(f"MODEL_STEP In Batch {batch_index}, batch data count: {batch_data.count()}")

                optim_host_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_data, self.model_weights, self.encrypted_calculator, self.optimizer, self.n_iter_,
                    batch_index)
                LOGGER.debug('optim_host_gradient: {}'.format(optim_host_gradient))

                self.gradient_loss_operator.compute_loss(self.model_weights, self.optimizer, self.n_iter_, batch_index)

                # clip gradient
                if self.model_param.clip_gradient and self.model_param.clip_gradient > 0:
                    optim_host_gradient = np.maximum(optim_host_gradient, -self.model_param.clip_gradient)
                    optim_host_gradient = np.minimum(optim_host_gradient, self.model_param.clip_gradient)

                _model_weights = self.optimizer.update_model(self.model_weights, optim_host_gradient)
                self.model_weights.update(_model_weights)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))

            LOGGER.info("Get is_converged flag from arbiter:{}".format(self.is_converged))

            validation_strategy.validate(self, self.n_iter_)

            self.n_iter_ += 1
            LOGGER.info("iter: {}, is_converged: {}".format(self.n_iter_, self.is_converged))
            if self.is_converged:
                break

        LOGGER.debug("Final fm weights: {}".format(self.model_weights.unboxed))

    def predict(self, data_instances):
        """
        Prediction of fm
        Parameters
        ----------
        data_instances:DTable of Instance, input data
        """
        LOGGER.info("Start predict ...")
        if self.need_one_vs_rest:
            self.one_vs_rest_obj.predict(data_instances)
            return

        prob_host = self.compute_fm(data_instances, self.model_weights)
        vx_host = self.compute_vx(data_instances, self.model_weights.embed_)
        prob_host = prob_host.join(vx_host, lambda a, b: (a, b))

        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")
