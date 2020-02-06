#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from arch.api.utils import log_utils
from federatedml.util import consts
from federatedml.optim.convergence import converge_func_factory
from fate_flow.entity.metric import MetricType, MetricMeta, Metric
from federatedrec.general_mf.hetero_gmf.hetero_gmf_base import HeteroGMFBase

LOGGER = log_utils.getLogger()


class HeteroGMFArbiter(HeteroGMFBase):
    def __init__(self):
        super(HeteroGMFArbiter, self).__init__()
        self.role = consts.ARBITER

    def _init_model(self, params):
        super()._init_model(params)
        early_stop = self.model_param.early_stop
        self.converge_func = converge_func_factory(early_stop.converge_func, early_stop.eps).is_converge
        self.loss_consumed = early_stop.converge_func != "weight_diff"

    def callback_loss(self, iter_num, loss):
        """
        call back function of loss and metrics.
        :param iter_num: iter number.
        :param loss: loss type.
        :return:
        """
        metric_meta = MetricMeta(name='train',
                                 metric_type=MetricType.LOSS,
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])

    def _check_monitored_status(self):
        loss = self.aggregator.aggregate_loss(suffix=self._iter_suffix())
        LOGGER.info(f"loss at iter {self.aggregator_iter}: {loss}")
        self.callback_loss(self.aggregator_iter, loss)
        if self.loss_consumed:
            converge_args = (loss,) if self.loss_consumed else (self.aggregator.model,)
            return self.aggregator.send_converge_status(self.converge_func,
                                                        converge_args=converge_args,
                                                        suffix=self._iter_suffix())

    def fit(self, data_inst):
        """
        Aggregate model for host and guest, then broadcast back.
        :param data_inst: input param is not used.
        :return:
        """
        while self.aggregator_iter < self.max_iter:
            self.aggregator.aggregate_and_broadcast(suffix=self._iter_suffix())

            if self._check_monitored_status():
                LOGGER.info(f"early stop at iter {self.aggregator_iter}")
                break
            self.aggregator_iter += 1
        else:
            LOGGER.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def save_model(self):
        """
        :return:aggregated model.
        """
        return self.aggregator.model
