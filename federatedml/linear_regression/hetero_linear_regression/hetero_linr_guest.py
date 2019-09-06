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

from operator import add

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.model_selection import MiniBatch
from federatedml.optim.gradient import HeteroLinearGradient
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic import data_overview
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRGuest(HeteroLinRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []

        self.guest_loss = None
        self.guest_wxy = None
        self.wx = None
        self.role = consts.GUEST

    def compute_intermediate(self, data_instances, coef_, intercept_):
        wx = self.compute_wx(data_instances, coef_, intercept_)
        wxy = wx.join(data_instances, lambda wx, d: wx - d.label)
        return

    def aggregate_loss(self, host_forward_wx, host_forward_loss):
        """
        Compute total loss = loss_host + loss_guest + loss_hg, where
        loss_hg = 2 * [[wx_h]] * (wx_g - y)
        Parameters
        ----------
        host_forward_wx: DTable, include encrypted W * X
        host_forward_loss: encrypted loss computed by host

        Returns
        ----------
        aggregate_loss
        """

        hg_loss = self.guest_wxy.join(host_forward_wx,
                                      lambda g, h: (g * h))
        hg_loss = 2 * hg_loss.reduce(add)
        aggregate_loss = self.guest_loss + host_forward_loss + hg_loss
        return aggregate_loss

    def fit(self, data_instances):
        """
        Train linr model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_linr_guest fit")
        self._abnormal_detection(data_instances)

        self.header = self.get_header(data_instances)

        public_key = federation.get(name=self.transfer_variable.paillier_pubkey.name,
                                    tag=self.transfer_variable.generate_transferid(
                                        self.transfer_variable.paillier_pubkey),
                                    idx=0)
        LOGGER.info("Get public_key from arbiter:{}".format(public_key))
        self.encrypt_operator.set_public_key(public_key)

        LOGGER.info("Generate mini-batch from input data")
        mini_batch_obj = MiniBatch(data_instances, batch_size=self.batch_size)
        batch_num = mini_batch_obj.batch_nums
        if self.batch_size == -1:
            LOGGER.info("batch size is -1, set it to the number of data in data_instances")
            self.batch_size = data_instances.count()

        batch_info = {"batch_size": self.batch_size, "batch_num": batch_num}
        LOGGER.info("batch_info:{}".format(batch_info))
        federation.remote(batch_info,
                          name=self.transfer_variable.batch_info.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                          role=consts.HOST,
                          idx=0)
        LOGGER.info("Remote batch_info to host")
        federation.remote(batch_info,
                          name=self.transfer_variable.batch_info.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                          role=consts.ARBITER,
                          idx=0)
        LOGGER.info("Remote batch_info to arbiter")

        self.encrypted_calculator = [EncryptModeCalculator(self.encrypt_operator,
                                                           self.encrypted_mode_calculator_param.mode,
                                                           self.encrypted_mode_calculator_param.re_encrypted_rate) for _
                                     in range(batch_num)]

        LOGGER.info("Start initialize model.")
        # LOGGER.info("fit_intercept:{}".format(self.init_param_obj.fit_intercept))
        model_shape = self.get_features_shape(data_instances)
        weight = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        if self.init_param_obj.fit_intercept is True:
            self.coef_ = weight[:-1]
            self.intercept_ = weight[-1]
        else:
            self.coef_ = weight

        is_send_all_batch_index = False
        self.n_iter_ = 0
        index_data_inst_map = {}

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            # each iter will get the same batach_data_generator
            batch_data_generator = mini_batch_obj.mini_batch_data_generator(result='index')

            batch_index = 0
            for batch_data_index in batch_data_generator:
                LOGGER.info("batch:{}".format(batch_index))
                if not is_send_all_batch_index:
                    LOGGER.info("remote mini-batch index to Host")
                    federation.remote(batch_data_index,
                                      name=self.transfer_variable.batch_data_index.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.batch_data_index,
                                          self.n_iter_,
                                          batch_index),
                                      role=consts.HOST,
                                      idx=0)
                    if batch_index >= mini_batch_obj.batch_nums - 1:
                        is_send_all_batch_index = True

                # Get mini-batch train data
                if len(index_data_inst_map) < batch_num:
                    batch_data_inst = data_instances.join(batch_data_index, lambda data_inst, index: data_inst)
                    index_data_inst_map[batch_index] = batch_data_inst
                else:
                    batch_data_inst = index_data_inst_map[batch_index]

                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data_inst)

                # host forward
                if self.gradient_operator is None:
                    self.gradient_operator = HeteroLinearGradient(
                        self.encrypt_operator)
                self.compute_intermediate(batch_feat_inst, self.coef_, self.intercept_)
                host_forward_wx = federation.get(name=self.transfer_variable.host_forward_wx.name,
                                                 tag=self.transfer_variable.generate_transferid(
                                                     self.transfer_variable.host_forward_wx, self.n_iter_, batch_index),
                                                 idx=0)
                LOGGER.info("Get host_forward_wx from host")

                host_forward_loss = federation.get(
                    name=self.transfer_variable.host_forward_loss.name,
                    tag=self.transfer_variable.generate_transferid(
                        self.transfer_variable.host_forward_loss, self.n_iter_,
                        batch_index),
                    idx=0)
                LOGGER.info("Get host_forward_loss from host")

                # compute [[d]]

                residual = self.gradient_operator.compute_residual(batch_feat_inst, self.wx, host_forward_wx)
                federation.remote(residual,
                                  name=self.transfer_variable.residual.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.residual,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.HOST,
                                  idx=0)

                LOGGER.info("Remote residual to host")
                # compute guest gradient and loss
                guest_gradient = self.gradient_operator.compute_gradient(batch_feat_inst,
                                                                         residual,
                                                                         self.fit_intercept)

                federation.remote(guest_gradient,
                                  name=self.transfer_variable.guest_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_gradient,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote guest_gradient to arbiter")

                optim_guest_gradient = federation.get(name=self.transfer_variable.optim_guest_gradient.name,
                                                      tag=self.transfer_variable.generate_transferid(
                                                          self.transfer_variable.optim_guest_gradient, self.n_iter_,
                                                          batch_index),
                                                      idx=0)
                LOGGER.info("Get optim_guest_gradient from arbiter")

                # update model
                LOGGER.info("update_model")
                self.update_model(optim_guest_gradient)

                loss = self.aggregate_loss(host_forward_wx, host_forward_loss)
                # loss regulation if necessary
                if self.updater is not None:
                    guest_loss_regular = self.updater.loss_norm(self.coef_)
                    loss += self.encrypt_operator.encrypt(guest_loss_regular)

                federation.remote(loss,
                                  name=self.transfer_variable.loss.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.loss,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote loss to arbiter")

                # is converge of loss in arbiter
                batch_index += 1

                # temporary resource recovery and will be removed in the future
                rubbish_list = [host_forward_wx,
                                residual,
                                self.guest_wxy
                                ]
                data_overview.rubbish_clear(rubbish_list)

            is_stopped = federation.get(name=self.transfer_variable.is_stopped.name,
                                        tag=self.transfer_variable.generate_transferid(
                                            self.transfer_variable.is_stopped, self.n_iter_, batch_index),
                                        idx=0)
            LOGGER.info("Get is_stopped flag from arbiter:{}".format(is_stopped))

            self.n_iter_ += 1
            if is_stopped:
                LOGGER.info("Get stop signal from arbiter, model is converged, iter:{}".format(self.n_iter_))
                break
        LOGGER.info("guest model coef: {}, intercept {}".format(self.coef_, self.intercept_))
        LOGGER.info("Reach max iter {}, train model finish!".format(self.max_iter))

    def predict(self, data_instances):
        """
        Prediction of linear regression
        Parameters
        ----------
        data_instances:DTable of Instance, input data

        Returns
        ----------
        DTable
            include prediction result
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)
        guest_partial_prediction = self.compute_wx(data_features, self.coef_, self.intercept_)
        host_partial_prediction = federation.get(name=self.transfer_variable.host_partial_prediction.name,
                                                 tag=self.transfer_variable.generate_transferid(
                                                     self.transfer_variable.host_partial_prediction),
                                                 idx=0)
        LOGGER.info("Get partial predictions from host")

        prediction = guest_partial_prediction.join(host_partial_prediction, lambda g, h: g + h)
        predict_result = data_instances.join(prediction,
                                             lambda inst, pred: [inst.label, float(pred),
                                                                 float(pred), {"label": round(pred)}])
        return predict_result
