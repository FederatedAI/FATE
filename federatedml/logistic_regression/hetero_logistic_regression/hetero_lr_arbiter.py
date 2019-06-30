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
from fate_flow.entity.metric import MetricMeta, Metric
from federatedml.logistic_regression.hetero_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim.federated_aggregator import HeteroFederatedAggregator
from federatedml.util.transfer_variable.hetero_lr_transfer_variable import HeteroLRTransferVariable
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLRArbiter(HeteroLRBase):
    def __init__(self):
        # LogisticParamChecker.check_param(logistic_params)
        super(HeteroLRArbiter, self).__init__()

        # attribute
        self.pre_loss = None
        self.batch_num = None
        self.transfer_variable = HeteroLRTransferVariable()

    def perform_subtasks(self, **training_info):
        """
        performs any tasks that the arbiter is responsible for.

        This 'perform_subtasks' function serves as a handler on conducting any task that the arbiter is responsible
        for. For example, for the 'perform_subtasks' function of 'HeteroDNNLRArbiter' class located in
        'hetero_dnn_lr_arbiter.py', it performs some works related to updating/training local neural networks of guest
        or host.

        For this particular class (i.e., 'HeteroLRArbiter') that serves as a base arbiter class for neural-networks-based
        hetero-logistic-regression model, the 'perform_subtasks' function will do nothing. In other words, no subtask is
        performed by this arbiter.

        :param training_info: a dictionary holding training information
        """
        pass

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)

        if not "model" in args:
            LOGGER.info("Task is fit")
            self.fit()
        else:
            LOGGER.info("Task is transform")

    def fit(self, data_instances=None):
        """
        Train lr model of role arbiter
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        """

        LOGGER.info("Enter hetero_lr_arbiter fit")
        if data_instances:
            # self.header = data_instance.schema.get('header')
            self.header = self.get_header(data_instances)
        else:
            self.header = []

        # Generate encrypt keys
        self.encrypt_operator.generate_key(self.key_length)
        public_key = self.encrypt_operator.get_public_key()
        public_key = public_key
        LOGGER.info("public_key:{}".format(public_key))
        federation.remote(public_key,
                          name=self.transfer_variable.paillier_pubkey.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                          role=consts.HOST,
                          idx=0)
        LOGGER.info("remote public_key to host")

        federation.remote(public_key,
                          name=self.transfer_variable.paillier_pubkey.name,
                          tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                          role=consts.GUEST,
                          idx=0)
        LOGGER.info("remote public_key to guest")

        batch_info = federation.get(name=self.transfer_variable.batch_info.name,
                                    tag=self.transfer_variable.generate_transferid(self.transfer_variable.batch_info),
                                    idx=0)
        LOGGER.info("Get batch_info from guest:{}".format(batch_info))
        self.batch_num = batch_info["batch_num"]

        is_stop = False
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            LOGGER.info("iter:{}".format(self.n_iter_))
            batch_index = 0
            iter_loss = 0
            while batch_index < self.batch_num:
                LOGGER.info("batch:{}".format(batch_index))
                host_gradient = federation.get(name=self.transfer_variable.host_gradient.name,
                                               tag=self.transfer_variable.generate_transferid(
                                                   self.transfer_variable.host_gradient, self.n_iter_, batch_index),
                                               idx=0)
                LOGGER.info("Get host_gradient from Host")

                guest_gradient = federation.get(name=self.transfer_variable.guest_gradient.name,
                                                tag=self.transfer_variable.generate_transferid(
                                                    self.transfer_variable.guest_gradient, self.n_iter_, batch_index),
                                                idx=0)
                LOGGER.info("Get guest_gradient from Guest")

                # aggregate gradient
                host_gradient, guest_gradient = np.array(host_gradient), np.array(guest_gradient)
                gradient = np.hstack((host_gradient, guest_gradient))
                # decrypt gradient
                for i in range(gradient.shape[0]):
                    gradient[i] = self.encrypt_operator.decrypt(gradient[i])

                # optimization
                optim_gradient = self.optimizer.apply_gradients(gradient)
                # separate optim_gradient according gradient size of Host and Guest
                separate_optim_gradient = HeteroFederatedAggregator.separate(optim_gradient,
                                                                             [host_gradient.shape[0],
                                                                              guest_gradient.shape[0]])
                host_optim_gradient = separate_optim_gradient[0]
                guest_optim_gradient = separate_optim_gradient[1]

                federation.remote(host_optim_gradient,
                                  name=self.transfer_variable.host_optim_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.host_optim_gradient,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.HOST,
                                  idx=0)
                LOGGER.info("Remote host_optim_gradient to Host")

                federation.remote(guest_optim_gradient,
                                  name=self.transfer_variable.guest_optim_gradient.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.guest_optim_gradient,
                                      self.n_iter_,
                                      batch_index),
                                  role=consts.GUEST,
                                  idx=0)
                LOGGER.info("Remote guest_optim_gradient to Guest")

                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.perform_subtasks(**training_info)

                loss = federation.get(name=self.transfer_variable.loss.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.loss, self.n_iter_, batch_index),
                                      idx=0)

                de_loss = self.encrypt_operator.decrypt(loss)
                iter_loss += de_loss
                # LOGGER.info("Get loss from guest:{}".format(de_loss))

                batch_index += 1

            # if converge
            loss = iter_loss / self.batch_num
            LOGGER.info("iter loss:{}".format(loss))
            metric_meta = MetricMeta(name='train',
                                     metric_type="LOSS",
                                     extra_metas={
                                         "unit_name": "hetero_lr"
                                     })
            metric_name = 'loss_' + self.flowid
            self.callback_meta(metric_name=metric_name, metric_namespace='train', metric_meta=metric_meta)
            self.callback_metric(metric_name=metric_name,
                                 metric_namespace='train',
                                 metric_data=[Metric(self.n_iter_, loss)])
            if self.converge_func.is_converge(loss):
                is_stop = True

            federation.remote(is_stop,
                              name=self.transfer_variable.is_stopped.name,
                              tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_stopped,
                                                                             self.n_iter_,
                                                                             batch_index),
                              role=consts.HOST,
                              idx=0)
            LOGGER.info("Remote is_stop to host:{}".format(is_stop))

            federation.remote(is_stop,
                              name=self.transfer_variable.is_stopped.name,
                              tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_stopped,
                                                                             self.n_iter_,
                                                                             batch_index),
                              role=consts.GUEST,
                              idx=0)
            LOGGER.info("Remote is_stop to guest:{}".format(is_stop))

            self.n_iter_ += 1
            if is_stop:
                LOGGER.info("Model is converged, iter:{}".format(self.n_iter_))
                break

        LOGGER.info("Reach max iter {} or converge, train model finish!".format(self.max_iter))
