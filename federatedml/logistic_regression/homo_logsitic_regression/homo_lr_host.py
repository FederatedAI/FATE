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

import functools

import numpy as np

from arch.api import federation
from arch.api.utils import log_utils
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.model_selection import MiniBatch
from federatedml.optim import Initializer
from federatedml.optim import activation
from federatedml.optim.federated_aggregator.homo_federated_aggregator import HomoFederatedAggregator
from federatedml.optim.gradient import LogisticGradient, TaylorLogisticGradient
from federatedml.statistic import data_overview
from federatedml.util import consts
from fate_flow.entity.metric import MetricMeta
from fate_flow.entity.metric import Metric
from federatedml.util.transfer_variable.homo_lr_transfer_variable import HomoLRTransferVariable
from arch.api.proto import lr_model_param_pb2

LOGGER = log_utils.getLogger()


class HomoLRHost(HomoLRBase):
    def __init__(self):
        super(HomoLRHost, self).__init__()

        self.aggregator = HomoFederatedAggregator()

        self.transfer_variable = HomoLRTransferVariable()
        self.initializer = Initializer()
        self.mini_batch_obj = None
        self.classes_ = [0, 1]
        self.has_sychronized_encryption = False
        self.role = consts.HOST

    def _init_model(self, params):
        super(HomoLRHost, self)._init_model(params)
        encrypt_params = params.encrypt_param
        if encrypt_params.method in [consts.PAILLIER]:
            self.use_encrypt = True
        else:
            self.use_encrypt = False

        if self.use_encrypt and params.penalty == 'L1':
            raise RuntimeError("Encrypted homo-lr supports L2 penalty or 'none' only")

        if self.use_encrypt:
            self.gradient_operator = TaylorLogisticGradient()
            self.re_encrypt_batches = params.re_encrypt_batches
        else:
            self.gradient_operator = LogisticGradient()

    def fit(self, data_instances):
        if not self.need_run:
            return data_instances

        self.header = data_instances.schema.get('header')  # ['x1', 'x2', 'x3' ... ]

        self._abnormal_detection(data_instances)

        self.__init_parameters(data_instances)

        w = self.__init_model(data_instances)

        for iter_num in range(self.max_iter):
            # mini-batch
            LOGGER.debug("In iter: {}".format(iter_num))
            batch_data_generator = self.mini_batch_obj.mini_batch_data_generator()
            batch_num = 0
            total_loss = 0

            for batch_data in batch_data_generator:
                f = functools.partial(self.gradient_operator.compute,
                                      coef=self.coef_,
                                      intercept=self.intercept_,
                                      fit_intercept=self.fit_intercept)

                grad_loss = batch_data.mapPartitions(f)

                n = batch_data.count()
                if not self.use_encrypt:
                    grad, loss = grad_loss.reduce(self.aggregator.aggregate_grad_loss)
                    grad = np.array(grad)
                    grad /= n
                    loss /= n
                    if self.updater is not None:
                        loss_norm = self.updater.loss_norm(self.coef_)
                        total_loss += loss + loss_norm

                    metric_meta = MetricMeta(name='train',
                                             metric_type="LOSS",
                                             extra_metas={
                                                 "unit_name": "homo_lr"
                                             })
                    self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
                    self.callback_metric(metric_name='loss',
                                         metric_namespace='train',
                                         metric_data=[Metric(iter_num, total_loss)])

                else:
                    grad, _ = grad_loss.reduce(self.aggregator.aggregate_grad)
                    grad = np.array(grad)
                    grad /= n

                self.update_model(grad)
                w = self.merge_model()

                batch_num += 1
                if self.use_encrypt and batch_num % self.re_encrypt_batches == 0:
                    to_encrypt_model_id = self.transfer_variable.generate_transferid(
                        self.transfer_variable.to_encrypt_model, iter_num, batch_num
                    )

                    federation.remote(w,
                                      name=self.transfer_variable.to_encrypt_model.name,
                                      tag=to_encrypt_model_id,
                                      role=consts.ARBITER,
                                      idx=0)

                    re_encrypted_model_id = self.transfer_variable.generate_transferid(
                        self.transfer_variable.re_encrypted_model, iter_num, batch_num
                    )
                    LOGGER.debug("re_encrypted_model_id: {}".format(re_encrypted_model_id))
                    w = federation.get(name=self.transfer_variable.re_encrypted_model.name,
                                       tag=re_encrypted_model_id,
                                       idx=0)

                    w = np.array(w)
                    self.set_coef_(w)

            model_transfer_id = self.transfer_variable.generate_transferid(
                self.transfer_variable.host_model, iter_num)
            federation.remote(w,
                              name=self.transfer_variable.host_model.name,
                              tag=model_transfer_id,
                              role=consts.ARBITER,
                              idx=0)

            if not self.use_encrypt:
                loss_transfer_id = self.transfer_variable.generate_transferid(
                    self.transfer_variable.host_loss, iter_num)

                federation.remote(total_loss,
                                  name=self.transfer_variable.host_loss.name,
                                  tag=loss_transfer_id,
                                  role=consts.ARBITER,
                                  idx=0)

            LOGGER.debug("model and loss sent")

            final_model_id = self.transfer_variable.generate_transferid(
                self.transfer_variable.final_model, iter_num)

            w = federation.get(name=self.transfer_variable.final_model.name,
                               tag=final_model_id,
                               idx=0)

            w = np.array(w)
            self.set_coef_(w)

            converge_flag_id = self.transfer_variable.generate_transferid(
                self.transfer_variable.converge_flag, iter_num)

            converge_flag = federation.get(name=self.transfer_variable.converge_flag.name,
                                           tag=converge_flag_id,
                                           idx=0)

            self.n_iter_ = iter_num
            LOGGER.debug("converge_flag: {}".format(converge_flag))
            if converge_flag:
                break
                # self.save_model()
        self.data_output = data_instances

    def __init_parameters(self, data_instances):

        party_weight_id = self.transfer_variable.generate_transferid(
            self.transfer_variable.host_party_weight
        )
        LOGGER.debug("Start to remote party_weight: {}, transfer_id: {}".format(self.party_weight, party_weight_id))

        federation.remote(self.party_weight,
                          name=self.transfer_variable.host_party_weight.name,
                          tag=party_weight_id,
                          role=consts.ARBITER,
                          idx=0)

        self.__synchronize_encryption()

        # Send re-encrypt times
        self.mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)
        if self.use_encrypt:
            # LOGGER.debug("Use encryption, send re_encrypt_times")
            total_batch_num = self.mini_batch_obj.batch_nums
            re_encrypt_times = total_batch_num // self.re_encrypt_batches
            transfer_id = self.transfer_variable.generate_transferid(self.transfer_variable.re_encrypt_times)
            LOGGER.debug("Start to remote re_encrypt_times: {}, transfer_id: {}".format(re_encrypt_times, transfer_id))

            federation.remote(re_encrypt_times,
                              name=self.transfer_variable.re_encrypt_times.name,
                              tag=transfer_id,
                              role=consts.ARBITER,
                              idx=0)
            LOGGER.info("sent re_encrypt_times: {}".format(re_encrypt_times))

    def __synchronize_encryption(self):
        """
        Communicate with hosts. Specify whether use encryption or not and transfer the public keys.
        """
        # Send if this host use encryption or not
        use_encryption_id = self.transfer_variable.generate_transferid(
            self.transfer_variable.use_encrypt
        )
        LOGGER.debug("Start to remote use_encrypt: {}, transfer_id: {}".format(self.use_encrypt, use_encryption_id))

        federation.remote(self.use_encrypt,
                          name=self.transfer_variable.use_encrypt.name,
                          tag=use_encryption_id,
                          role=consts.ARBITER,
                          idx=0)

        # Set public key
        if self.use_encrypt:
            pubkey_id = self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey)
            pubkey = federation.get(name=self.transfer_variable.paillier_pubkey.name,
                                    tag=pubkey_id,
                                    idx=0)
            LOGGER.debug("Received pubkey")
            self.encrypt_operator.set_public_key(pubkey)
        LOGGER.info("Finish synchronized ecryption")
        self.has_sychronized_encryption = True

    def predict(self, data_instances):
        if not self.need_run:
            return data_instances

        if not self.has_sychronized_encryption:
            self.__synchronize_encryption()
            self.__load_arbiter_model()
        else:
            LOGGER.info("in predict, has synchronize encryption information")

        from federatedml.statistic.data_overview import get_features_shape
        feature_shape = get_features_shape(data_instances)
        LOGGER.debug("Shape of coef_ : {}, feature shape: {}".format(len(self.coef_), feature_shape))

        wx = self.compute_wx(data_instances, self.coef_, self.intercept_)

        if self.use_encrypt:
            encrypted_wx_id = self.transfer_variable.generate_transferid(self.transfer_variable.predict_wx)
            LOGGER.debug("Host encrypted wx id: {}".format(encrypted_wx_id))
            LOGGER.debug("Start to remote wx: {}, transfer_id: {}".format(wx, encrypted_wx_id))
            federation.remote(wx,
                              name=self.transfer_variable.predict_wx.name,
                              tag=encrypted_wx_id,
                              role=consts.ARBITER,
                              idx=0)
            predict_result_id = self.transfer_variable.generate_transferid(self.transfer_variable.predict_result)
            predict_result = federation.get(name=self.transfer_variable.predict_result.name,
                                            tag=predict_result_id,
                                            idx=0)
            # local_predict_table = predict_result.collect()
            predict_result_table = predict_result.join(data_instances, lambda p, d: [d.label, None, p])
        else:
            pred_prob = wx.mapValues(lambda x: activation.sigmoid(x))
            pred_label = self.classified(pred_prob, self.predict_param.threshold)
            if self.predict_param.with_proba:
                predict_result = data_instances.mapValues(lambda x: x.label)
                predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
            else:
                predict_result = data_instances.mapValues(lambda x: (x.label, None))
            predict_result_table = predict_result.join(pred_label, lambda x, y: [x[0], x[1], y])
        LOGGER.debug("Finish predict")
        return predict_result_table

    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param):
        return {}

    def __init_model(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)

        w = self.encrypt_operator.encrypt_list(w)
        w = np.array(w)

        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0
        return w

    def __load_arbiter_model(self):
        final_model_id = self.transfer_variable.generate_transferid(self.transfer_variable.final_model, "predict")
        final_model = federation.get(name=self.transfer_variable.final_model.name,
                                     tag=final_model_id,
                                     idx=0)
        LOGGER.info("Received arbiter's model")
        LOGGER.debug("final_model: {}".format(final_model))
        self.set_coef_(final_model)

    def _get_param(self):
        header = self.header
        weight_dict = {}
        for idx, header_name in enumerate(header):
            coef_i = self.coef_[idx]
            weight_dict[header_name] = coef_i

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(iters=self.n_iter_,
                                                             loss_history=[],
                                                             is_converged=self.is_converged,
                                                             weight={},
                                                             intercept=0,
                                                             header=header)
        from google.protobuf import json_format
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj
