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
from federatedml.hetero_lr_utils.procedure import aggregator, paillier_cipher, batch_generator
from federatedml.optim import activation
from federatedml.optim.gradient import HeteroLogisticGradient
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.statistic.data_overview import rubbish_clear
from federatedml.util import consts
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class HeteroLRGuest(HeteroLRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        self.guest_forward = None
        self.role = consts.GUEST
        self.aggregator = aggregator.Guest()
        self.cipher = paillier_cipher.Guest()
        self.batch_generator = batch_generator.Guest()

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

        en_wx_join_en_wx_square = en_wx.join(en_wx_square, lambda wx, wx_square: (wx, wx_square))
        self.guest_forward = en_wx_join_en_wx_square.join(wx, lambda e, wx: (e[0], e[1], wx))

        # temporary resource recovery and will be removed in the future
        rubbish_list = [ en_wx, wx_square, en_wx_square, en_wx_join_en_wx_square]
        rubbish_clear(rubbish_list)

    def aggregate_forward(self, host_forward):
        """
        Compute (en_wx_g + en_wx_h)^2 = en_wx_g^2 + en_wx_h^2 + 2 * wx_g * en_wx_h , where en_wx_g is the encrypted W * X + b of guest, wx_g is unencrypted W * X + b,
        and en_wx_h is the encrypted W * X + b of host.
        Parameters
        ----------
        host_forward: DTable, include encrypted W * X and (W * X)^2

        Returns
        ----------
        aggregate_forward_res
        list
            include W * X and (W * X)^2 federate with guest and host
        """
        aggregate_forward_res = self.guest_forward.join(host_forward,
                                                        lambda g, h: (g[0] + h[0], g[1] + h[1] + 2 * g[2] * h[0]))
        return aggregate_forward_res

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

    def fit(self, data_instances):
        """
        Train lr model of role guest
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_lr_guest fit")
        self._abnormal_detection(data_instances)
        self.header = self.get_header(data_instances)
        data_instances = data_instances.mapValues(HeteroLRGuest.load_data)

        self.cipher_operator = self.cipher.gen_paillier_cipher_operator()

        LOGGER.info("Generate mini-batch from input data")
        self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)

        # TODO: different encrypter
        # self.encrypted_calculator = [EncryptModeCalculator(self.cipher_operator,
        #                                                    self.encrypted_mode_calculator_param.mode,
        #                                                    self.encrypted_mode_calculator_param.re_encrypted_rate) for _
        #                              in range(batch_num)]

        LOGGER.info("Start initialize model.")
        LOGGER.info("fit_intercept:{}".format(self.init_param_obj.fit_intercept))
        model_shape = self.get_features_shape(data_instances)
        self.lr_variables = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            # each iter will get the same batach_data_generator
            batch_data_generator = self.batch_generator.generate_batch_data()

            batch_index = 0
            for batch_data in batch_data_generator:

                # transforms features of raw input 'batch_data_inst' into more representative features 'batch_feat_inst'
                batch_feat_inst = self.transform(batch_data)

                # guest/host forward
                self.compute_forward(batch_feat_inst, self.coef_, self.intercept_, batch_index)
                host_forward = federation.get(name=self.transfer_variable.host_forward_dict.name,
                                              tag=self.transfer_variable.generate_transferid(
                                                  self.transfer_variable.host_forward_dict, self.n_iter_, batch_index),
                                              idx=0)
                LOGGER.info("Get host_forward from host")
                aggregate_forward_res = self.aggregate_forward(host_forward)
                en_aggregate_wx = aggregate_forward_res.mapValues(lambda v: v[0])
                en_aggregate_wx_square = aggregate_forward_res.mapValues(lambda v: v[1])

                # compute [[d]]
                if self.gradient_operator is None:
                    self.gradient_operator = HeteroLogisticGradient(self.cipher_operator)
                fore_gradient = self.gradient_operator.compute_fore_gradient(batch_feat_inst, en_aggregate_wx)
                federation.remote(fore_gradient,
                                  name=self.transfer_variable.fore_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.fore_gradient,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.HOST,
                                  idx=0)

                LOGGER.info("Remote fore_gradient to Host")
                # compute guest gradient and loss
                guest_gradient, loss = self.gradient_operator.compute_gradient_and_loss(batch_feat_inst,
                                                                                        fore_gradient,
                                                                                        en_aggregate_wx,
                                                                                        en_aggregate_wx_square,
                                                                                        self.fit_intercept)

                # loss regulation if necessary
                if self.updater is not None:
                    guest_loss_regular = self.updater.loss_norm(self.coef_)
                    loss += self.cipher_operator.encrypt(guest_loss_regular)

                federation.remote(guest_gradient,
                                  name=self.transfer_variable.guest_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_gradient,
                                                                                 self.n_iter_,
                                                                                 batch_index),
                                  role=consts.ARBITER,
                                  idx=0)
                LOGGER.info("Remote guest_gradient to arbiter")

                optim_guest_gradient = federation.get(name=self.transfer_variable.guest_optim_gradient.name,
                                                      tag=self.transfer_variable.generate_transferid(
                                                          self.transfer_variable.guest_optim_gradient, self.n_iter_,
                                                          batch_index),
                                                      idx=0)
                LOGGER.info("Get optim_guest_gradient from arbiter")

                # update model
                LOGGER.info("update_model")
                self.update_model(optim_guest_gradient)

                # update local model that transforms features of raw input 'batch_data_inst'
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.update_local_model(fore_gradient, batch_data_inst, self.coef_, **training_info)

                # Get loss regulation from Host if regulation is set
                if self.updater is not None:
                    en_host_loss_regular = federation.get(name=self.transfer_variable.host_loss_regular.name,
                                                          tag=self.transfer_variable.generate_transferid(
                                                              self.transfer_variable.host_loss_regular, self.n_iter_,
                                                              batch_index),
                                                          idx=0)
                    LOGGER.info("Get host_loss_regular from Host")
                    loss += en_host_loss_regular

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
                rubbish_list = [host_forward,
                                aggregate_forward_res,
                                en_aggregate_wx,
                                en_aggregate_wx_square,
                                fore_gradient,
                                self.guest_forward
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
        predict_param: PredictParam, the setting of prediction.

        Returns
        ----------
        DTable
            include input data label, predict probably, label
        """
        LOGGER.info("Start predict ...")

        data_features = self.transform(data_instances)
        prob_guest = self.compute_wx(data_features, self.coef_, self.intercept_)
        prob_host = federation.get(name=self.transfer_variable.host_prob.name,
                                   tag=self.transfer_variable.generate_transferid(
                                       self.transfer_variable.host_prob),
                                   idx=0)
        LOGGER.info("Get probability from Host")

        # guest probability
        pred_prob = prob_guest.join(prob_host, lambda g, h: activation.sigmoid(g + h))
        pred_label = self.classified(pred_prob, self.predict_param.threshold)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"0": (1 - x[1]), "1": x[1]}])

        return predict_result
