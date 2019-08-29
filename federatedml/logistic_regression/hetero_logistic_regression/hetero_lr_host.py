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

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.logistic_regression.hetero_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim.gradient import HeteroLogisticGradient
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import consts
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class HeteroLRHost(HeteroLRBase):
    def __init__(self):
        super(HeteroLRHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []
        self.role = consts.HOST

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

    def fit(self, data_instances):
        """
        Train lr model of role host
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_lr host")
        self._abnormal_detection(data_instances)

        self.header = self.get_header(data_instances)
        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)

        # TODO: different encrypter
        # self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
        #                                                    self.encrypted_mode_calculator_param.mode,
        #                                                    self.encrypted_mode_calculator_param.re_encrypted_rate) for _
        #                              in range(batch_num)]

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)

        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False

        if self.fit_intercept:
            self.fit_intercept = False

        self.lr_variables = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))

            batch_data_generator = self.batch_generator.generate_batch_data()

            for batch_data in batch_data_generator:
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)

                # compute forward
                host_forward = self.compute_forward(batch_feat_inst, self.coef_, self.intercept_, batch_index)
                federation.remote(host_forward,
                                  name=self.transfer_variable.host_forward_dict.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_forward_dict,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote host_forward to guest")

                # compute host gradient
                fore_gradient = federation.get(name=self.transfer_variable.fore_gradient.name,
                                               tag=self.transfer_variable.generate_transferid(
                                                   self.transfer_variable.fore_gradient, self.n_iter_, batch_index),
                                               idx=0)
                LOGGER.info("Get fore_gradient from guest")
                if self.gradient_operator is None:
                    self.gradient_operator = HeteroLogisticGradient(self.cipher_operator)
                host_gradient = self.gradient_operator.compute_gradient(batch_feat_inst, fore_gradient,
                                                                        fit_intercept=False)
                # regulation if necessary
                if self.updater is not None:
                    loss_regular = self.updater.loss_norm(self.coef_)
                    en_loss_regular = self.cipher_operator.encrypt(loss_regular)
                    federation.remote(en_loss_regular,
                                      name=self.transfer_variable.host_loss_regular.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.host_loss_regular,
                                          self.n_iter_,
                                          batch_index),
                                      role=consts.GUEST,
                                      idx=0)
                    LOGGER.info("Remote host_loss_regular to guest")

                federation.remote(host_gradient,
                                  name=self.transfer_variable.host_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_gradient,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote host_gradient to arbiter")

                # Get optimize host gradient and update model
                optim_host_gradient = federation.get(name=self.transfer_variable.host_optim_gradient.name,
                                                     tag=self.transfer_variable.generate_transferid(
                                                         self.transfer_variable.host_optim_gradient, self.n_iter_,
                                                         batch_index),
                                                     idx=0)
                LOGGER.info("Get optim_host_gradient from arbiter")

                LOGGER.info("update_model")
                self.update_model(optim_host_gradient)

                # update local model that transforms features of raw input 'batch_data_inst'
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.update_local_model(fore_gradient, batch_data_inst, self.coef_, **training_info)

                batch_index += 1

                # temporary resource recovery and will be removed in the future
                rubbish_list = [host_forward,
                                fore_gradient
                                ]
                data_overview.rubbish_clear(rubbish_list)

            is_stopped = federation.get(name=self.transfer_variable.is_stopped.name,
                                        tag=self.transfer_variable.generate_transferid(
                                            self.transfer_variable.is_stopped, self.n_iter_, batch_index),
                                        idx=0)
            LOGGER.info("Get is_stop flag from arbiter:{}".format(is_stopped))

            self.n_iter_ += 1
            if is_stopped:
                LOGGER.info("Get stop signal from arbiter, model is converged, iter:{}".format(self.n_iter_))
                break

        LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))

    def predict(self, data_instances):
        """
        Prediction of lr
        Parameters
        ----------
        data_instances:DTable of Instance, input data
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)

        prob_host = self.compute_wx(data_features, self.coef_, self.intercept_)
        federation.remote(prob_host,
                          name=self.transfer_variable.host_prob.name,
                          tag=self.transfer_variable.generate_transferid(
                              self.transfer_variable.host_prob),
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("Remote probability to Guest")
