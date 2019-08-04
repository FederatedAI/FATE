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
from federatedml.evaluation import Evaluation
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_base import HomoLRBase
from federatedml.model_selection import MiniBatch
from federatedml.optim import Initializer
from federatedml.optim import activation
from federatedml.optim.federated_aggregator.homo_federated_aggregator import HomoFederatedAggregator
from federatedml.optim.gradient import LogisticGradient
from fate_flow.entity.metric import MetricMeta
from fate_flow.entity.metric import Metric
from federatedml.util import consts
from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class HomoLRGuest(HomoLRBase):
    def __init__(self):
        super(HomoLRGuest, self).__init__()
        self.aggregator = HomoFederatedAggregator
        self.gradient_operator = LogisticGradient()

        self.initializer = Initializer()
        self.classes_ = [0, 1]

        self.evaluator = Evaluation()
        self.loss_history = []
        self.is_converged = False
        self.role = consts.GUEST

    def fit(self, data_instances):
        if not self.need_run:
            return data_instances

        self._abnormal_detection(data_instances)
        self.init_schema(data_instances)
        self.__init_parameters()

        self.__init_model(data_instances)

        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=self.batch_size)

        for iter_num in range(self.max_iter):
            # mini-batch
            batch_data_generator = mini_batch_obj.mini_batch_data_generator()
            total_loss = 0
            batch_num = 0

            for batch_data in batch_data_generator:
                n = batch_data.count()

                f = functools.partial(self.gradient_operator.compute,
                                      coef=self.coef_,
                                      intercept=self.intercept_,
                                      fit_intercept=self.fit_intercept)
                grad_loss = batch_data.mapPartitions(f)

                grad, loss = grad_loss.reduce(self.aggregator.aggregate_grad_loss)

                grad /= n
                loss /= n

                if self.updater is not None:
                    loss_norm = self.updater.loss_norm(self.coef_)
                    total_loss += (loss + loss_norm)
                delta_grad = self.optimizer.apply_gradients(grad)

                self.update_model(delta_grad)
                batch_num += 1

            total_loss /= batch_num
            w = self.merge_model()
            metric_meta = MetricMeta(name='train',
                                     metric_type="LOSS",
                                     extra_metas={
                                         "unit_name": "iters"
                                     })
            # metric_name = self.get_metric_name('loss')
            flow_id_list = self.flowid.split('.')
            if len(flow_id_list) == 0:
                metric_namespace = 'train'
            else:
                metric_namespace = '.'.join(flow_id_list[1:])
            self.callback_meta(metric_name='loss', metric_namespace=metric_namespace, metric_meta=metric_meta)
            self.callback_metric(metric_name='loss',
                                 metric_namespace=metric_namespace,
                                 metric_data=[Metric(iter_num, total_loss)])

            self.loss_history.append(total_loss)
            LOGGER.info("iter: {}, loss: {}".format(iter_num, total_loss))
            # send model
            model_transfer_id = self.transfer_variable.generate_transferid(self.transfer_variable.guest_model,
                                                                           iter_num)
            LOGGER.debug("Start to remote model: {}, transfer_id: {}".format(w, model_transfer_id))

            federation.remote(w,
                              name=self.transfer_variable.guest_model.name,
                              tag=model_transfer_id,
                              role=consts.ARBITER,
                              idx=0)

            # send loss

            loss_transfer_id = self.transfer_variable.generate_transferid(self.transfer_variable.guest_loss, iter_num)
            LOGGER.debug("Start to remote total_loss: {}, transfer_id: {}".format(total_loss, loss_transfer_id))
            federation.remote(total_loss,
                              name=self.transfer_variable.guest_loss.name,
                              tag=loss_transfer_id,
                              role=consts.ARBITER,
                              idx=0)

            # recv model
            model_transfer_id = self.transfer_variable.generate_transferid(
                self.transfer_variable.final_model, iter_num)
            w = federation.get(name=self.transfer_variable.final_model.name,
                               tag=model_transfer_id,
                               idx=0)

            w = np.array(w)
            self.set_coef_(w)

            # recv converge flag
            converge_flag_id = self.transfer_variable.generate_transferid(self.transfer_variable.converge_flag,
                                                                          iter_num)
            converge_flag = federation.get(name=self.transfer_variable.converge_flag.name,
                                           tag=converge_flag_id,
                                           idx=0)

            self.n_iter_ = iter_num
            LOGGER.debug("converge flag is :{}".format(converge_flag))

            if converge_flag:
                self.is_converged = True
                break

    def __init_parameters(self):
        party_weight_id = self.transfer_variable.generate_transferid(
            self.transfer_variable.guest_party_weight
        )
        LOGGER.debug("Start to remote party_weight: {}, transfer_id: {}".format(self.party_weight, party_weight_id))
        federation.remote(self.party_weight,
                          name=self.transfer_variable.guest_party_weight.name,
                          tag=party_weight_id,
                          role=consts.ARBITER,
                          idx=0)

        # LOGGER.debug("party weight sent")
        LOGGER.info("Finish initialize parameters")

    def __init_model(self, data_instances):
        model_shape = data_overview.get_features_shape(data_instances)

        LOGGER.info("Initialized model shape is {}".format(model_shape))

        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
            self.intercept_ = 0

        # LOGGER.debug("Initialed model")
        return w

    def predict(self, data_instances):
        LOGGER.debug("Get in predict, data_instance count: {}, need_run: {}".format(data_instances.count(),
                                                                                    self.need_run))

        if not self.need_run:
            return data_instances
        LOGGER.debug("homo_lr guest need run predict, coef: {}, instercept: {}".format(len(self.coef_), self.intercept_))
        wx = self.compute_wx(data_instances, self.coef_, self.intercept_)
        pred_prob = wx.mapValues(lambda x: activation.sigmoid(x))
        pred_label = self.classified(pred_prob, self.predict_param.threshold)

        predict_result = data_instances.mapValues(lambda x: x.label)
        predict_result = predict_result.join(pred_prob, lambda x, y: (x, y))
        predict_result = predict_result.join(pred_label, lambda x, y: [x[0], y, x[1], {"1": x[1], "0": (1 - x[1])}])
        return predict_result
