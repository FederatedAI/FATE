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
from federatedml.logistic_regression.base_logistic_regression import BaseLogisticRegression
from federatedml.optim.gradient import HeteroLogisticGradient
from federatedml.util import consts
from federatedml.util import LogisticParamChecker
from federatedml.util.transfer_variable import HeteroLRTransferVariable
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroLRHost(BaseLogisticRegression):
    def __init__(self, logistic_params):
        LogisticParamChecker.check_param(logistic_params)
        
        super(HeteroLRHost, self).__init__(logistic_params)
        self.transfer_variable = HeteroLRTransferVariable()
        self.batch_num = None
        self.batch_index_list = []

    def compute_forward(self, data_instances, coef_, intercept_):
        wx = self.compute_wx(data_instances, coef_, intercept_)
        encrypt_operator = self.encrypt_operator
        host_forward = wx.mapValues(lambda v: (encrypt_operator.encrypt(v), encrypt_operator.encrypt(np.square(v))))
        return host_forward

    def fit(self, data_instances):
        LOGGER.info("Enter hetero_lr host")
        public_key = federation.get(name=self.transfer_variable.paillier_pubkey.name,
                                    tag=self.transfer_variable.generate_transferid(
                                        self.transfer_variable.paillier_pubkey),
                                    idx=0)

        LOGGER.info("Get public_key from arbiter:{}".format(public_key))
        self.encrypt_operator.set_public_key(public_key)

        batch_info = federation.get(name=self.transfer_variable.batch_info.name,
                                    tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                                    idx=0)
        LOGGER.info("Get batch_info from guest:" + str(batch_info))
        self.batch_size = batch_info["batch_size"]
        self.batch_num = batch_info["batch_num"]

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)

        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False

        if self.fit_intercept:
            self.fit_intercept = False

        self.coef_ = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        is_stopped = False
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))
            batch_index = 0
            while batch_index < self.batch_num:
                # set batch_data
                if len(self.batch_index_list) < self.batch_num:
                    batch_data_index = federation.get(name=self.transfer_variable.batch_data_index.name,
                                                      tag=self.transfer_variable.generate_transferid(
                                                          self.transfer_variable.batch_data_index, self.n_iter_,
                                                          batch_index),
                                                      idx=0)
                    LOGGER.info("Get batch_index from Guest")
                    self.batch_index_list.append(batch_data_index)
                else:
                    batch_data_index = self.batch_index_list[batch_index]

                # Get mini-batch train data
                batch_data_inst = batch_data_index.join(data_instances, lambda g, d: d)

                # compute forward
                host_forward = self.compute_forward(batch_data_inst, self.coef_, self.intercept_)
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
                    self.gradient_operator = HeteroLogisticGradient(self.encrypt_operator)
                host_gradient = self.gradient_operator.compute_gradient(data_instances, fore_gradient,
                                                                        fit_intercept=False)
                # regulation if necessary
                if self.updater is not None:
                    loss_regular = self.updater.loss_norm(self.coef_)
                    en_loss_regular = self.encrypt_operator.encrypt(loss_regular)
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

                # is converge
                is_stopped = federation.get(name=self.transfer_variable.is_stopped.name,
                                            tag=self.transfer_variable.generate_transferid(
                                                self.transfer_variable.is_stopped, self.n_iter_, batch_index),
                                            idx=0)
                LOGGER.info("Get is_stop flag from arbiter:{}".format(is_stopped))

                batch_index += 1
                if is_stopped:
                    LOGGER.info("Get stop signal from arbiter, model is converged, iter:{}".format(self.n_iter_))
                    break

            self.n_iter_ += 1
            if is_stopped:
                break

        LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))

    def predict(self, data_instances, predict_param=None):
        LOGGER.info("Start predict ...")
        prob_host = self.compute_wx(data_instances, self.coef_, self.intercept_)
        federation.remote(prob_host,
                          name=self.transfer_variable.host_prob.name,
                          tag=self.transfer_variable.generate_transferid(
                              self.transfer_variable.host_prob),
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("Remote probability to Host")
