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
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta, MetricType
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.optim import activation
from federatedml.optim.federated_aggregator import HomoFederatedAggregator
from federatedml.secureprotol import PaillierEncrypt, FakeEncrypt
from federatedml.homo.procedure.aggregate import Aggregate
from federatedml.homo.procedure.aggregated_dispatcher import AggregatedDisPatcher
from federatedml.homo.procedure.paillier_cipher import PaillierCipherProcedure
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HomoLRArbiter(HomoLRBase):
    def __init__(self):
        super(HomoLRArbiter, self).__init__()
        self.aggregator = HomoFederatedAggregator()

        self.classes_ = [0, 1]

        # To be initialized
        self.host_use_encryption = []
        self.re_encrypt_times = []  # Record the times needed for each host
        self.curt_re_encrypt_times = []
        self.host_encrypter = []
        self.party_weights = []  # The first one is guest weight, host weights for otherwise
        self.has_sychronized_encryption = False
        self.loss_history = []
        self.is_converged = False
        self.header = []
        self.role = consts.ARBITER

    def _init_model(self, params):
        super(HomoLRArbiter, self)._init_model(params)
        self.encrypt_param = params.encrypt_param

    def fit(self, data=None):
        if not self.need_run:
            return data

        LOGGER.debug("self.has_synchronized_encryption: {}".format(self.has_synchronized_encryption))
        if not self.has_synchronized_encryption:
            self.mean_aggregator = Aggregate.arbiter(self.transfer_variable)
            self.model_dispatcher = AggregatedDisPatcher.arbiter(self.transfer_variable)
            self.mean_aggregator.get_party_weights()
            self.paillier_cipher_procedure = PaillierCipherProcedure.arbiter(self.transfer_variable, "train")
            self.ciphers = self.paillier_cipher_procedure.maybe_gen_pubkey(self.encrypt_param.key_length)
            self.paillier_cipher_procedure.set_re_cipher_time()
            self.has_synchronized_encryption = True
            LOGGER.debug("self.has_synchronized_encryption: {}".format(self.has_synchronized_encryption))

        LOGGER.info("Finish init parameters")

        for iter_num in range(self.max_iter):

            # re_encrypt host models
            self.paillier_cipher_procedure.re_cipher(iter_num, self.re_encrypt_batches)

            # Aggregate models receive from each party
            final_model = self.mean_aggregator.aggregate(iter_num, cipher=self.ciphers)

            total_loss = self.aggregator.aggregate_loss(transfer_variable=self.transfer_variable,
                                                        iter_num=iter_num,
                                                        party_weights=self.party_weights,
                                                        host_use_encryption=self.host_use_encryption)

            self.loss_history.append(total_loss)

            if not self.need_one_vs_rest:
                metric_meta = MetricMeta(name='train',
                                         metric_type=MetricType.LOSS,
                                         extra_metas={
                                             "unit_name": "iters"
                                         })
                metric_name = self.get_metric_name('loss')
                self.callback_meta(metric_name=metric_name, metric_namespace='train', metric_meta=metric_meta)
                self.callback_metric(metric_name=metric_name,
                                     metric_namespace='train',
                                     metric_data=[Metric(iter_num, total_loss)])

            LOGGER.info("Iter: {}, loss: {}".format(iter_num, total_loss))
            # send model
            self.model_dispatcher.send(final_model, iter_num, ciphers=self.ciphers)

            if self.use_loss:
                converge_flag = self.converge_func.is_converge(total_loss)
            else:
                converge_flag = self.converge_func.is_converge(final_model)
            converge_flag_id = self.transfer_variable.generate_transferid(
                self.transfer_variable.converge_flag,
                iter_num)

            federation.remote(converge_flag,
                              name=self.transfer_variable.converge_flag.name,
                              tag=converge_flag_id,
                              role=consts.GUEST,
                              idx=0)
            federation.remote(converge_flag,
                              name=self.transfer_variable.converge_flag.name,
                              tag=converge_flag_id,
                              role=consts.HOST,
                              idx=-1)
            self.set_coef_(final_model)
            self.n_iter_ = iter_num
            if converge_flag:
                self.is_converged = True
                break
        self._set_header()
        self.data_output = data

    def predict(self, data=None):
        LOGGER.debug("In arbiter's predict, need run: {}".format(self.need_run))
        if not self.need_run:
            return data

        # synchronize encryption information
        if not self.has_synchronized_encryption:
            self.paillier_cipher_procedure = PaillierCipherProcedure.arbiter(self.transfer_variable, "predict")
            self.ciphers = self.paillier_cipher_procedure.maybe_gen_pubkey(self.encrypt_param.key_length)
            self.has_synchronized_encryption = True
            model = self.merge_model()
            AggregatedDisPatcher.arbiter(self.transfer_variable).send(model, "predict", self.ciphers)

        for idx, use_encrypt in enumerate(self.host_use_encryption):
            if use_encrypt:
                encrypter = self.host_encrypter[idx]
                predict_wx_id = self.transfer_variable.generate_transferid(self.transfer_variable.predict_wx)
                LOGGER.debug("Arbiter encrypted wx id: {}".format(predict_wx_id))

                predict_wx = federation.get(name=self.transfer_variable.predict_wx.name,
                                            tag=predict_wx_id,
                                            idx=idx
                                            )
                decrypted_wx = encrypter.distribute_decrypt(predict_wx)
                pred_prob = decrypted_wx.mapValues(lambda x: activation.sigmoid(x))
                pred_label = self.classified(pred_prob, self.predict_param.threshold)
                predict_result_id = self.transfer_variable.generate_transferid(self.transfer_variable.predict_result)
                LOGGER.debug("predict_result_id: {}".format(predict_result_id))

                LOGGER.debug(
                    "Start to remote pred_label: {}, transfer_id: {}".format(pred_label, predict_result_id))
                federation.remote(pred_label,
                                  name=self.transfer_variable.predict_result.name,
                                  tag=predict_result_id,
                                  role=consts.HOST,
                                  idx=idx)
        LOGGER.info("Finish predicting, result has been sent back")
        return

    def _set_header(self):
        self.header = ['head_' + str(x) for x in range(len(self.coef_))]

    def run(self, component_parameters=None, args=None):
        """
        Rewrite run function so that it can start fit and predict without input data.
        """
        self._init_runtime_parameters(component_parameters)
        data_sets = args["data"]

        need_eval = False
        for data_key in data_sets:

            if "eval_data" in data_sets[data_key]:
                need_eval = True
            else:
                need_eval = False

        if self.need_cv:
            self.cross_validation(None)
        elif self.need_one_vs_rest:
            if "model" in args:
                self._load_model(args)
                self.one_vs_rest_predict(None)
            else:
                self.one_vs_rest_fit()
                self.data_output = self.one_vs_rest_predict(None)
                if need_eval:
                    self.data_output = self.one_vs_rest_predict(None)
        elif "model" in args:
            self._load_model(args)
            self.set_flowid('predict')
            self.predict()
        else:
            self.set_flowid('train')
            self.fit()
            self.set_flowid('predict')
            self.data_output = self.predict()

            if need_eval:
                self.set_flowid('validate')
                self.predict()
