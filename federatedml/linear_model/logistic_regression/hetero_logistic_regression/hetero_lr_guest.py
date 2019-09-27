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
from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim import activation
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.secureprotol import EncryptModeCalculator

from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLRGuest(HeteroLRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        # self.guest_forward = None
        self.role = consts.GUEST
        self.cipher = paillier_cipher.Guest()
        self.batch_generator = batch_generator.Guest()
        self.gradient_loss_operator = hetero_lr_gradient_and_loss.Guest()
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
        Train lr model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_lr_guest fit")
        self._abnormal_detection(data_instances)
        self.header = self.get_header(data_instances)
       
        validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        
        data_instances = data_instances.mapValues(HeteroLRGuest.load_data)

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
        self.model_weights = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            # each iter will get the same batach_data_generator
            batch_data_generator = self.batch_generator.generate_batch_data()
            self.optimizer.set_iters(self.n_iter_)
            batch_index = 0
            for batch_data in batch_data_generator:
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)

                # Start gradient procedure

                optim_guest_gradient, fore_gradient, host_forwards = self.gradient_loss_operator. \
                    compute_gradient_procedure(
                        batch_feat_inst,
                        self.encrypted_calculator,
                        self.model_weights,
                        self.optimizer,
                        self.n_iter_,
                        batch_index
                        )
                LOGGER.debug('optim_guest_gradient: {}'.format(optim_guest_gradient))
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.update_local_model(fore_gradient, data_instances, self.model_weights.coef_, **training_info)

                loss_norm = self.optimizer.loss_norm(self.model_weights)
                self.gradient_loss_operator.compute_loss(data_instances, self.n_iter_, batch_index, loss_norm)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_guest_gradient)
                batch_index += 1
                LOGGER.debug("lr_weight, iters: {}, update_model: {}".format(self.n_iter_, self.model_weights.unboxed))

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))
            LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
           
            validation_strategy.validate(self, self.n_iter_)
            
            self.n_iter_ += 1
            if self.is_converged:
                break

        LOGGER.debug("Final lr weights: {}".format(self.model_weights.unboxed))

    def predict(self, data_instances):
        """
        Prediction of lr
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
        prob_guest = self.compute_wx(data_features, self.model_weights.coef_, self.model_weights.intercept_)
        prob_host = self.transfer_variable.host_prob.get(idx=0)

        LOGGER.info("Get probability from Host")

        # guest probability
        pred_prob = prob_guest.join(prob_host, lambda g, h: activation.sigmoid(g + h))
        pred_label = pred_prob.mapValues(lambda x: 1 if x > self.model_param.predict_param.threshold else 0)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"0": (1 - x[1]), "1": x[1]}])

        return predict_result
