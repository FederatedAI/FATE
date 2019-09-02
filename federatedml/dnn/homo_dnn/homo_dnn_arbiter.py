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

from federatedml.framework.homo import random_padding_cipher_flow
from federatedml.model_base import ModelBase


class HomeDNNArbiter(ModelBase):

    def __init__(self):
        super().__init__()
        self.algo_flow = random_padding_cipher_flow.arbiter()

    def _init_model(self, model):
        self.max_iter = model.max_iter

    def fit(self, data=None):
        if not self.need_run:
            return data

        self.algo_flow.initialize()

        for iter_num in range(self.max_iter):

            self.algo_flow.aggregate(suffix=(iter_num,))

            # Part3: Aggregate models receive from each party
            final_model = self.aggregator.aggregate_model(transfer_variable=self.transfer_variable,
                                                          iter_num=iter_num,
                                                          party_weights=self.party_weights,
                                                          host_encrypter=self.host_encrypter)
            total_loss = self.aggregator.aggregate_loss(transfer_variable=self.transfer_variable,
                                                        iter_num=iter_num,
                                                        party_weights=self.party_weights,
                                                        host_use_encryption=self.host_use_encryption)
            # else:
            #     total_loss = np.linalg.norm(final_model)

            self.loss_history.append(total_loss)

            if not self.need_one_vs_rest:
                metric_meta = MetricMeta(name='train',
                                         metric_type="LOSS",
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
            final_model_id = self.transfer_variable.generate_transferid(self.transfer_variable.final_model, iter_num)
            # LOGGER.debug("Sending final_model, model id: {}, final_model: {}".format(final_model_id, final_model))
            federation.remote(final_model,
                              name=self.transfer_variable.final_model.name,
                              tag=final_model_id,
                              role=consts.GUEST,
                              idx=0)
            for idx, encrypter in enumerate(self.host_encrypter):
                encrypted_model = encrypter.encrypt_list(final_model)

                federation.remote(encrypted_model,
                                  name=self.transfer_variable.final_model.name,
                                  tag=final_model_id,
                                  role=consts.HOST,
                                  idx=idx)

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
