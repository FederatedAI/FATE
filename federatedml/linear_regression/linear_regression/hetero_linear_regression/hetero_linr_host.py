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

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim.gradient import HeteroLinearGradient
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRHost(HeteroLinRBase):
    def __init__(self):
        super(HeteroLinRHost, self).__init__()
        self.batch_num = None
        self.batch_index_list = []

    def compute_forward(self, data_instances, coef_, intercept_, batch_index=-1):
        wx = self.compute_wx(data_instances, coef_, intercept_)
        en_wx = self.encrypted_calculator[batch_index].encrypt(wx)
        # temporary resource recovery and will be removed in the future
        rubbish_list = [wx
                        ]
        rubbish_clear(rubbish_list)
        return en_wx

    def fit(self, data_instances):
        """
        Train linear regression model of role host
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_linr host")
        self._abnormal_detection(data_instances)

        self.header = self.get_header(data_instances)
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
        if self.batch_size < consts.MIN_BATCH_SIZE and self.batch_size != -1:
            raise ValueError(
                "Batch size get from guest should not less than 10, except -1, batch_size is {}".format(
                    self.batch_size))

        self.encrypted_calculator = [EncryptModeCalculator(self.encrypt_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(self.batch_num)]

        LOGGER.info("Start initialize model.")
        model_shape = self.get_features_shape(data_instances)
        # host does not hold intercept
        if self.init_param_obj.fit_intercept:
            self.init_param_obj.fit_intercept = False

        if self.fit_intercept:
            self.fit_intercept = False

        self.coef_ = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        self.n_iter_ = 0
        index_data_inst_map = {}

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:" + str(self.n_iter_))
            batch_index = 0
            while batch_index < self.batch_num:
                LOGGER.info("batch:{}".format(batch_index))
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
                if len(index_data_inst_map) < self.batch_num:
                    batch_data_inst = batch_data_index.join(data_instances, lambda g, d: d)
                    index_data_inst_map[batch_index] = batch_data_inst
                else:
                    batch_data_inst = index_data_inst_map[batch_index]

                LOGGER.info("batch_data_inst size:{}".format(batch_data_inst.count()))
                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data_inst)

                # compute forward
                host_forward_wx = self.compute_forward(batch_feat_inst, self.coef_, self.intercept_, batch_index)
                federation.remote(host_forward_wx,
                                  name=self.transfer_variable.host_forward_wx.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_forward_wx,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote host_forward_wx to guest")
                if self.gradient_operator is None:
                    self.gradient_operator = HeteroLinearGradient(self.encrypt_operator)
                loss = self.gradient_operator.compute_loss(batch_feat_inst, self.coef_, self.intercept_)

                if self.updater is not None:
                    loss_regular = self.updater.loss_norm(self.coef_)
                    loss = loss + loss_regular
                host_forward_loss = self.encrypt_operator.encrypt(loss)
                federation.remote(host_forward_loss,
                                  name=self.transfer_variable.host_forward_loss.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_forward_loss,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote host_forward_loss to guest")

                # compute host gradient
                residual = federation.get(name=self.transfer_variable.residual.name,
                                               tag=self.transfer_variable.generate_transferid(
                                                   self.transfer_variable.residual, self.n_iter_, batch_index),
                                               idx=0)
                LOGGER.info("Get residual from guest")

                host_gradient = self.gradient_operator.compute_gradient(batch_feat_inst, residual,
                                                                        fit_intercept=False)

                federation.remote(host_gradient,
                                  name=self.transfer_variable.host_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_gradient,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote host_gradient to arbiter")

                # Get decrypted host gradient and update model
                optim_gradient = federation.get(name=self.transfer_variable.optim_host_gradient.name,
                                                     tag=self.transfer_variable.generate_transferid(
                                                         self.transfer_variable.optim_host_gradient, self.n_iter_,
                                                         batch_index),
                                                     idx=0)
                LOGGER.info("Get optim_host_gradient from arbiter")

                LOGGER.info("update_model")
                self.update_model(optim_gradient)

                batch_index += 1

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
        Prediction of linear regression
        Parameters
        ----------
        data_instances:DTable of Instance, input data
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)

        partial_prediction = self.compute_wx(data_features, self.coef_, self.intercept_)
        federation.remote(partial_prediction,
                          name=self.transfer_variable.host_partial_prediction.name,
                          tag=self.transfer_variable.generate_transferid(
                              self.transfer_variable.host_partial_prediction),
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("Remote partial_prediction to Guest")
