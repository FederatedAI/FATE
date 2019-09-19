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
import typing

from arch.api.utils.log_utils import LoggerFactory
from federatedml.framework.homo.procedure import aggregator
from federatedml.model_base import ModelBase
from federatedml.nn.homo_nn import nn_model
from federatedml.optim.convergence import converge_func_factory
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.transfer_variable.transfer_class.homo_transfer_variable import HomoTransferVariable
from federatedml.util import consts

Logger = LoggerFactory.get_logger()


class HomoNNBase(ModelBase):

    def __init__(self):
        super().__init__()
        self.model_param = HomoNNParam()
        self.role = None

    def _init_model(self, param):
        super()._init_model(param)

        self.transfer_variable = HomoTransferVariable()
        secure_aggregate = param.secure_aggregate
        self.aggregator = aggregator.with_role(role=self.role,
                                               transfer_variable=self.transfer_variable,
                                               enable_secure_aggregate=secure_aggregate)
        self.max_iter = param.max_iter
        self.aggregator_iter = 0
        self.enable_converge_check = param.enable_converge_check

    def _iter_suffix(self):
        return self.aggregator_iter,


class HomoNNArbiter(HomoNNBase):

    def __init__(self):
        super().__init__()
        self.role = consts.ARBITER

    def _init_model(self, param):
        super(HomoNNArbiter, self)._init_model(param)
        self.converge_func = converge_func_factory(param).is_converge
        self.loss_consumed = param.converge_func != "weight_diff"

    def _check_converge_status(self):
        loss = self.aggregator.aggregate_loss(suffix=self._iter_suffix())
        Logger.info(f"loss at iter {self.aggregator_iter}: {loss}")
        if self.loss_consumed:
            converge_args = (loss,) if self.loss_consumed else (self.aggregator.model,)
            return self.aggregator.send_converge_status(self.converge_func,
                                                        converge_args=converge_args,
                                                        suffix=self._iter_suffix())

    def fit(self, data_inst):
        while self.aggregator_iter < self.max_iter:
            self.aggregator.aggregate_and_broadcast(suffix=self._iter_suffix())

            if self.enable_converge_check and self._check_converge_status():
                Logger.info(f"converged at iter {self.aggregator_iter}")
                break
            self.aggregator_iter += 1
        else:
            Logger.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def save_model(self):
        return self.aggregator.model


class HomoNNClient(HomoNNBase):

    def __init__(self):
        super().__init__()

    def _init_model(self, param):
        super(HomoNNClient, self)._init_model(param)

        nn_define = param.nn_define
        config_type = param.config_type
        self.nn_model, self.data_converter = nn_model.parse(config_type=config_type,
                                                            nn_define=nn_define,
                                                            optimizer=param.optimizer,
                                                            loss=param.loss,
                                                            metrics=param.metrics)
        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1

    def _check_converge_status(self, data, epoch_degree):
        metrics = self.nn_model.evaluate(data)
        Logger.info(f"metrics at iter {self.aggregator_iter}: {metrics}")
        loss = metrics["loss"]
        self.aggregator.send_loss(loss=loss,
                                  degree=epoch_degree,
                                  suffix=self._iter_suffix())
        return self.aggregator.get_converge_status(suffix=self._iter_suffix())

    def fit(self, data_inst):
        data = self.data_converter.convert(data_inst, batch_size=self.batch_size)
        epoch_degree = float(len(data))

        while self.aggregator_iter < self.max_iter:
            Logger.info(f"start {self.aggregator_iter}_th aggregation")

            # train
            self.nn_model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

            # send model for aggregate, then set aggregated model to local
            modify_func: typing.Callable = functools.partial(self.aggregator.aggregate_then_get,
                                                             degree=epoch_degree * self.aggregate_every_n_epoch,
                                                             suffix=self._iter_suffix())
            self.nn_model.modify(modify_func)

            # calc loss and check convergence
            if self.enable_converge_check and self._check_converge_status(data, epoch_degree):
                Logger.info(f"converged at iter {self.aggregator_iter}")
                break

            Logger.info(f"role {self.role} finish {self.aggregator_iter}_th aggregation")
            self.aggregator_iter += 1
        else:
            Logger.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def save_model(self):
        return self.nn_model.save_model()


class HomoNNHost(HomoNNClient):

    def __init__(self):
        super().__init__()
        self.role = consts.HOST


class HomoNNGuest(HomoNNClient):

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
