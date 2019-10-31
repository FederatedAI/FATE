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
import types
import typing
from functools import reduce

from arch.api.utils import log_utils
from federatedml.framework.homo.procedure import random_padding_cipher
from federatedml.framework.homo.sync import model_scatter_sync, \
    loss_transfer_sync, model_broadcast_sync, is_converge_sync
from federatedml.framework.weights import Weights, NumericWeights
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Arbiter(object):
    def __init__(self):
        self._model_scatter = None
        self._model_broadcaster = None
        self._loss_sync = None
        self._converge_sync = None
        self.model = None

    def register_aggregator(self, transfer_variables, enable_secure_aggregate=True):

        if enable_secure_aggregate:
            random_padding_cipher.Arbiter().register_random_padding_cipher(transfer_variables).exchange_secret_keys()

        self._model_scatter = model_scatter_sync.Arbiter().register_model_scatter(
            host_model_transfer=transfer_variables.host_model,
            guest_model_transfer=transfer_variables.guest_model)

        self._model_broadcaster = model_broadcast_sync.Arbiter(). \
            register_model_broadcaster(model_transfer=transfer_variables.aggregated_model)

        self._loss_sync = loss_transfer_sync.Arbiter().register_loss_transfer(
            host_loss_transfer=transfer_variables.host_loss,
            guest_loss_transfer=transfer_variables.guest_loss)

        self._converge_sync = is_converge_sync.Arbiter().register_is_converge(
            is_converge_variable=transfer_variables.is_converge)

        return self

    def aggregate_model(self, ciphers_dict=None, suffix=tuple()) -> Weights:
        models = self.get_models_for_aggregate(ciphers_dict, suffix=suffix)
        total_model, total_degree = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), models)
        total_model /= total_degree
        LOGGER.debug("In aggregate model, total_model: {}, total_degree: {}".format(total_model.unboxed, total_degree))
        return total_model

    def aggregate_and_broadcast(self, ciphers_dict=None, suffix=tuple()):
        """
        aggregate models from guest and hosts, then broadcast the aggregated model.

        Args:
            ciphers_dict: a dict of host id to host cipher
            suffix: tag suffix
        """
        self.model = self.aggregate_model(ciphers_dict=ciphers_dict, suffix=suffix)
        self.send_aggregated_model(self.model, ciphers_dict=ciphers_dict, suffix=suffix)
        return self.model

    def get_models_for_aggregate(self, ciphers_dict=None, suffix=tuple()):
        return self._model_scatter.get_models(ciphers_dict=ciphers_dict, suffix=suffix)

    def send_aggregated_model(self, model: Weights, ciphers_dict=None, suffix=tuple()):
        self._model_broadcaster.send_model(model=model, ciphers_dict=ciphers_dict, suffix=suffix)

    def aggregate_loss(self, idx=None, suffix=tuple()):
        losses = self._loss_sync.get_losses(idx=idx, suffix=suffix)
        total_loss = 0.0
        total_degree = 0.0
        for loss in losses:
            total_loss += loss.unboxed
            total_degree += loss.get_degree(1.0)
        return total_loss / total_degree

    def send_converge_status(self, converge_func: types.FunctionType, converge_args, suffix=tuple()):
        return self._converge_sync.check_converge_status(converge_func=converge_func, converge_args=converge_args,
                                                         suffix=suffix)


class Client(object):
    def __init__(self):
        self._secure_aggregate_cipher = None
        self._model_scatter = None
        self._model_broadcaster = None
        self._loss_sync = None
        self._converge_sync = None
        self._enable_secure_aggregate = False

    def secure_aggregate(self, send_func, weights: Weights, degree: float = None, enable_secure_aggregate=True):
        # w -> w * degree
        if degree:
            weights *= degree
        # w * degree -> w * degree + \sum(\delta(i, j) * r_{ij}), namely， adding random mask.
        if enable_secure_aggregate:
            weights = weights.encrypted(cipher=self._secure_aggregate_cipher, inplace=True)
        # maybe remote degree
        remote_weights = weights.for_remote().with_degree(degree) if degree else weights.for_remote()

        send_func(remote_weights)

    def send_model(self, weights: Weights, degree: float = None, suffix=tuple()):
        return self.secure_aggregate(send_func=functools.partial(self._model_scatter.send_model, suffix=suffix),
                                     weights=weights, degree=degree,
                                     enable_secure_aggregate=self._enable_secure_aggregate)

    def get_aggregated_model(self, suffix=tuple()):
        return self._model_broadcaster.get_model(suffix=suffix)

    def aggregate_then_get(self, model: Weights, degree: float = None, suffix=tuple()) -> Weights:
        self.send_model(weights=model, degree=degree, suffix=suffix)
        return self.get_aggregated_model(suffix=suffix)

    def send_loss(self, loss: typing.Union[float, Weights], degree: float = None, suffix=tuple()):
        if isinstance(loss, float):
            loss = NumericWeights(loss)
        return self.secure_aggregate(send_func=functools.partial(self._loss_sync.send_loss, suffix=suffix),
                                     weights=loss, degree=degree,
                                     enable_secure_aggregate=False)

    def get_converge_status(self, suffix=tuple()):
        return self._converge_sync.get_converge_status(suffix=suffix)


class Guest(Client):
    def register_aggregator(self, transfer_variables, enable_secure_aggregate=True):
        self._enable_secure_aggregate = enable_secure_aggregate
        if enable_secure_aggregate:
            self._secure_aggregate_cipher = random_padding_cipher.Guest().register_random_padding_cipher(
                transfer_variables).create_cipher()

        self._model_scatter = model_scatter_sync.Guest().register_model_scatter(
            model_transfer=transfer_variables.guest_model)

        self._model_broadcaster = model_broadcast_sync.Guest(). \
            register_model_broadcaster(model_transfer=transfer_variables.aggregated_model)

        self._loss_sync = loss_transfer_sync.Guest().register_loss_transfer(
            loss_transfer=transfer_variables.guest_loss)

        self._converge_sync = is_converge_sync.Guest().register_is_converge(
            is_converge_variable=transfer_variables.is_converge)

        return self


class Host(Client):
    def register_aggregator(self, transfer_variables, enable_secure_aggregate=True):
        self._enable_secure_aggregate = enable_secure_aggregate
        if enable_secure_aggregate:
            self._secure_aggregate_cipher = random_padding_cipher.Host().register_random_padding_cipher(
                transfer_variables).create_cipher()

        self._model_scatter = model_scatter_sync.Host().register_model_scatter(
            model_transfer=transfer_variables.host_model)

        self._model_broadcaster = model_broadcast_sync.Host(). \
            register_model_broadcaster(model_transfer=transfer_variables.aggregated_model)

        self._loss_sync = loss_transfer_sync.Host().register_loss_transfer(
            loss_transfer=transfer_variables.host_loss)

        self._converge_sync = is_converge_sync.Host().register_is_converge(
            is_converge_variable=transfer_variables.is_converge)

        return self


def with_role(role, transfer_variable, enable_secure_aggregate=True):
    if role == consts.GUEST:
        return Guest().register_aggregator(transfer_variable, enable_secure_aggregate)
    elif role == consts.HOST:
        return Host().register_aggregator(transfer_variable, enable_secure_aggregate)
    elif role == consts.ARBITER:
        return Arbiter().register_aggregator(transfer_variable, enable_secure_aggregate)
    else:
        raise ValueError(f"role {role} not found")
