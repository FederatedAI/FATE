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

import numpy as np

from arch.api import federation
from arch.api.utils import log_utils
from fate_flow.entity.metric import MetricMeta, Metric
from federatedml.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim.federated_aggregator import HeteroFederatedAggregator
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRArbiter(HeteroLinRBase):
    def __init__(self):
        super().__init__()
        super(HeteroLinRArbiter, self).__init__()

        # attribute
        self.pre_loss = None
        self.batch_num = None

    def run(self, component_parameters=None, args=None):
        """
        check mode of task
        :param component_parameters: for cross validation
        :param args: string, task input
        :return:
        """
        self._init_runtime_parameters(component_parameters)
        need_cv = self.need_cv

        if need_cv:
            LOGGER.info("Task is cross validation")
            self.cross_validation(None)
            return
        if "model" not in args:
            LOGGER.info("Task is fit")
            self.set_flowid('train')
            self.fit()
        else:
            LOGGER.info("Task is transform")

    def fit(self, data_instances=None):
        """
        Train linear regression model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """
        LOGGER.info("Enter hetero_linr_arbiter fit")
        if data_instances:
            self.header = self.get_header(data_instances)
        else:
            self.header = []

        # Generate encrypt key pairs
        self.encrypt_operator.generate_key(self.key_length)
        public_key = self.encrypt_operator.get_public_key()
        LOGGER.info("public_key:{}".format(public_key))

        federation.remote(public_key,
                          name=self.transfer_variable.paillier_pubkey.name,
                          tag=self.transfer_variable.generate_transferid(
                              self.transfer_variable.paillier_pubkey),
                          role=consts.HOST,
                          idx=0)
        LOGGER.info("remote public_key to host")

        federation.remote(public_key,
                          name=self.transfer_variable.paillier_pubkey.name,
                          tag=self.transfer_variable.generate_transferid(
                              self.transfer_variable.paillier_pubkey),
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("remote public_key to guest")

        batch_info = federation.get(name=self.transfer_variable.batch_info.name,
                                    tag=self.transfer_variable.generate_transferid(
                                        self.transfer_variable.batch_info),
                                    idx=0)
        LOGGER.info("Get batch_info from guest:{}".format(batch_info))
        self.batch_num = batch_info["batch_num"]

        is_stopped = False
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            batch_index = 0
            iter_loss = 0
            while batch_index < self.batch_num:
                LOGGER.info("batch:{}".format(batch_index))

                # receive gradients from Host & Guest
                host_gradient = federation.get(
                    name=self.transfer_variable.host_gradient.name,
                    tag=self.transfer_variable.generate_transferid(
                        self.transfer_variable.host_gradient, self.n_iter_,
                        batch_index),
                    idx=0)
                LOGGER.info("Get host_gradient from host")

                guest_gradient = federation.get(
                    name=self.transfer_variable.guest_gradient.name,
                    tag=self.transfer_variable.generate_transferid(
                        self.transfer_variable.guest_gradient, self.n_iter_,
                        batch_index),
                    idx=0)
                LOGGER.info("Get guest_gradient from guest")

                host_gradient, guest_gradient = np.array(
                    host_gradient), np.array(guest_gradient)
                gradient = np.hstack((host_gradient, guest_gradient))
                # decrypt gradient
                for i in range(gradient.shape[0]):
                    gradient[i] = self.encrypt_operator.decrypt(gradient[i])
                optim_gradient = self.optimizer.apply_gradients(gradient)
                # separate optim_gradient according gradient size of Host and Guest
                separate_optim_gradient = HeteroFederatedAggregator.separate(
                    optim_gradient,
                    [host_gradient.shape[0],
                     guest_gradient.shape[0]])
                optim_host_gradient = separate_optim_gradient[0]
                optim_guest_gradient = separate_optim_gradient[1]

                # send decrypted gradients back to Host & Guest
                federation.remote(optim_host_gradient,
                                  name=self.transfer_variable.optim_host_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.optim_host_gradient,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.HOST,
                                  idx=0)
                LOGGER.info("Remote optim_host_gradient to host")

                federation.remote(optim_guest_gradient,
                                  name=self.transfer_variable.optim_guest_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.optim_guest_gradient,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote optim_guest_gradient to guest")

                # receive loss from Guest
                loss = federation.get(name=self.transfer_variable.loss.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.loss,
                                          self.n_iter_, batch_index),
                                      idx=0)
                de_loss = self.encrypt_operator.decrypt(loss)
                iter_loss += de_loss / self.batch_size
                LOGGER.info("Get loss from guest:{}".format(de_loss))

                batch_index += 1
            # if converge
            loss = iter_loss / self.batch_num
            LOGGER.info("Total loss of this iteration:{}".format(loss))
            metric_meta = MetricMeta(name='train',
                                     metric_type='LOSS',
                                     extra_metas={
                                         "unit_name":"iters"
                                     })
            metric_name = 'loss_' + self.flowid
            self.callback_meta(metric_name=metric_name,
                               metric_namespace='train',
                               metric_meta=metric_meta)
            self.callback_metric(metric_name=metric_name,
                               metric_namespace='train',
                               metric_data=[Metric(self.n_iter_, float(loss))])

            if self.converge_func.is_converge(loss):
                is_stopped = True

            federation.remote(is_stopped,
                              name=self.transfer_variable.is_stopped.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.is_stopped,
                                  self.n_iter_,
                                  batch_index),
                              role=consts.HOST,
                              idx=0)
            LOGGER.info("Remote is_stopped to host:{}".format(is_stopped))

            federation.remote(is_stopped,
                              name=self.transfer_variable.is_stopped.name,
                              tag=self.transfer_variable.generate_transferid(
                                  self.transfer_variable.is_stopped,
                                  self.n_iter_,
                                  batch_index),
                              role=consts.GUEST,
                              idx=0)
            LOGGER.info("Remote is_stopped to guest:{}".format(is_stopped))

            self.n_iter_ += 1
            if is_stopped:
                LOGGER.info("Model is converged, iter:{}".format(self.n_iter_))
                break

        LOGGER.info("Reach max iter {} or converge, train model finish!".format(
            self.max_iter))
