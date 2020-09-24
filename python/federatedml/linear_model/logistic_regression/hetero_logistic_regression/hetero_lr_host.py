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

from federatedml.framework.hetero.procedure import convergence
from federatedml.framework.hetero.procedure import paillier_cipher, batch_generator
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroLRHost(HeteroLRBase):
    def __init__(self):
        super(HeteroLRHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []
        self.role = consts.HOST

        self.cipher = paillier_cipher.Host()
        self.batch_generator = batch_generator.Host()
        self.gradient_loss_operator = hetero_lr_gradient_and_loss.Host()
        self.converge_procedure = convergence.Host()
        self.encrypted_calculator = None

    def compute_forward(self, data_instances, coef_, intercept_, batch_index=-1):
        """
        Compute W * X + b and (W * X + b)^2, where X is the input data, W is the coefficient of lr,
        and b is the interception
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        coef_: list, coefficient of lr
        intercept_: float, the interception of lr
        """
        wx = self.compute_wx(data_instances, coef_, intercept_)

        en_wx = self.encrypted_calculator[batch_index].encrypt(wx)
        wx_square = wx.mapValues(lambda v: np.square(v))
        en_wx_square = self.encrypted_calculator[batch_index].encrypt(wx_square)

        host_forward = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))

        # temporary resource recovery and will be removed in the future
        rubbish_list = [wx,
                        en_wx,
                        wx_square,
                        en_wx_square
                        ]
        rubbish_clear(rubbish_list)

        return host_forward

    def fit(self, data_instances, validate_data=None):
        """
        Train lr model of role host
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_logistic_regression host")
        self.header = self.get_header(data_instances)

        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data):
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)
        self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        LOGGER.debug(f"MODEL_STEP Start fin_binary, data count: {data_instances.count()}")

        self.header = self.get_header(data_instances)
        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        self.batch_generator.initialize_batch_generator(data_instances)
        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_nums)

        self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_generator.batch_nums)]

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)
        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        # LOGGER.debug("model_shape: {}, w shape: {}, w: {}".format(model_shape, w.shape, w))
        self.model_weights = LinearModelWeights(w, fit_intercept=self.init_param_obj.fit_intercept)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))
            batch_data_generator = self.batch_generator.generate_batch_data()
            batch_index = 0
            self.optimizer.set_iters(self.n_iter_)
            for batch_data in batch_data_generator:
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)
                LOGGER.debug(f"MODEL_STEP In Batch {batch_index}, batch data count: {batch_feat_inst.count()}")

                optim_host_gradient, fore_gradient = self.gradient_loss_operator.compute_gradient_procedure(
                    batch_feat_inst, self.encrypted_calculator, self.model_weights, self.optimizer, self.n_iter_,
                    batch_index)
                # LOGGER.debug('optim_host_gradient: {}'.format(optim_host_gradient))

                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.update_local_model(fore_gradient, data_instances, self.model_weights.coef_, **training_info)

                self.gradient_loss_operator.compute_loss(self.model_weights, self.optimizer,
                                                         self.n_iter_, batch_index, self.cipher_operator)

                self.model_weights = self.optimizer.update_model(self.model_weights, optim_host_gradient)
                batch_index += 1

            self.is_converged = self.converge_procedure.sync_converge_info(suffix=(self.n_iter_,))

            LOGGER.info("Get is_converged flag from arbiter:{}".format(self.is_converged))

            if self.validation_strategy:
                LOGGER.debug('LR host running validation')
                self.validation_strategy.validate(self, self.n_iter_)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break
            self.n_iter_ += 1
            LOGGER.info("iter: {}, is_converged: {}".format(self.n_iter_, self.is_converged))
            if self.is_converged:
                break
        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)
        self.set_summary(self.get_model_summary())

        # LOGGER.debug("Final lr weights: {}".format(self.model_weights.unboxed))

    def predict(self, data_instances):
        self.transfer_variable.host_prob.disable_auto_clean()
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            self.one_vs_rest_obj.predict(data_instances)
            return

        data_features = self.transform(data_instances)

        prob_host = self.compute_wx(data_features, self.model_weights.coef_, self.model_weights.intercept_)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")
