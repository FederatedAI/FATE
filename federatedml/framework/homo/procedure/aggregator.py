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
import typing

import functools
import types
from functools import reduce

from arch.api.utils import log_utils
from federatedml.framework.homo.blocks import has_converged, loss_scatter, model_scatter, model_broadcaster
from federatedml.framework.homo.blocks import random_padding_cipher
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.has_converged import HasConvergedTransVar
from federatedml.framework.homo.blocks.loss_scatter import LossScatterTransVar
from federatedml.framework.homo.blocks.model_broadcaster import ModelBroadcasterTransVar
from federatedml.framework.homo.blocks.model_scatter import ModelScatterTransVar
from federatedml.framework.homo.blocks.random_padding_cipher import RandomPaddingCipherTransVar
from federatedml.framework.weights import Weights, NumericWeights, TransferableWeights
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class LegacyAggregatorTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST,), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.loss_scatter = LossScatterTransVar(server=server, clients=clients, prefix=self.prefix)
        self.has_converged = HasConvergedTransVar(server=server, clients=clients, prefix=self.prefix)
        self.model_scatter = ModelScatterTransVar(server=server, clients=clients, prefix=self.prefix)
        self.model_broadcaster = ModelBroadcasterTransVar(server=server, clients=clients, prefix=self.prefix)
        self.random_padding_cipher = RandomPaddingCipherTransVar(server=server, clients=clients, prefix=self.prefix)


class Arbiter(object):

    def __init__(self, trans_var=LegacyAggregatorTransVar()):
        self._guest_parties = trans_var.get_parties(roles=[consts.GUEST])
        self._host_parties = trans_var.get_parties(roles=[consts.HOST])
        self._client_parties = trans_var.client_parties

        self._loss_sync = loss_scatter.Server(trans_var.loss_scatter)
        self._converge_sync = has_converged.Server(trans_var.has_converged)
        self._model_scatter = model_scatter.Server(trans_var.model_scatter)
        self._model_broadcaster = model_broadcaster.Server(trans_var.model_broadcaster)
        self._random_padding_cipher = random_padding_cipher.Server(trans_var.random_padding_cipher)

    # noinspection PyUnusedLocal,PyAttributeOutsideInit,PyProtectedMember
    def register_aggregator(self, transfer_variables: BaseTransferVariables, enable_secure_aggregate=True):
        if enable_secure_aggregate:
            self._random_padding_cipher.exchange_secret_keys()
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
        model = self.aggregate_model(ciphers_dict=ciphers_dict, suffix=suffix)
        self.send_aggregated_model(model, ciphers_dict=ciphers_dict, suffix=suffix)
        return model

    def get_models_for_aggregate(self, ciphers_dict=None, suffix=tuple()):
        models = self._model_scatter.get_models(suffix=suffix)
        guest_model = models[0]
        yield (guest_model.weights, guest_model.get_degree() or 1.0)

        # host model
        index = 0
        for model in models[1:]:
            weights = model.weights
            if ciphers_dict and ciphers_dict.get(index, None):
                weights = weights.decrypted(ciphers_dict[index])
            yield (weights, model.get_degree() or 1.0)
            index += 1

    def send_aggregated_model(self, model: Weights,
                              ciphers_dict: typing.Union[None, typing.Mapping[int, typing.Any]] = None,
                              suffix=tuple()):
        if ciphers_dict is None:
            ciphers_dict = {}
        party_to_cipher = {self._host_parties[idx]: cipher for idx, cipher in ciphers_dict.items()}
        for party in self._client_parties:
            cipher = party_to_cipher.get(party)
            if cipher is None:
                self._model_broadcaster.send_model(model=model.for_remote(), parties=party, suffix=suffix)
            else:
                self._model_broadcaster.send_model(model=model.encrypted(cipher, False).for_remote(), parties=party,
                                                   suffix=suffix)

    def aggregate_loss(self, idx=None, suffix=tuple()):
        if idx is None:
            parties = None
        else:
            parties = []
            parties.extend(self._guest_parties)
            parties.extend([self._host_parties[i] for i in idx])
        losses = self._loss_sync.get_losses(parties=parties, suffix=suffix)
        total_loss = 0.0
        total_degree = 0.0
        for loss in losses:
            total_loss += loss.unboxed
            total_degree += loss.get_degree(1.0)
        return total_loss / total_degree

    def send_converge_status(self, converge_func: types.FunctionType, converge_args, suffix=tuple()):
        is_converge = converge_func(*converge_args)
        return self._converge_sync.remote_converge_status(is_converge, suffix=suffix)


class Client(object):
    def __init__(self, trans_var=LegacyAggregatorTransVar()):
        self._enable_secure_aggregate = False

        self._loss_sync = loss_scatter.Client(trans_var.loss_scatter)
        self._converge_sync = has_converged.Client(trans_var.has_converged)
        self._model_scatter = model_scatter.Client(trans_var.model_scatter)
        self._model_broadcaster = model_broadcaster.Client(trans_var.model_broadcaster)
        self._random_padding_cipher = random_padding_cipher.Client(trans_var.random_padding_cipher)

    # noinspection PyAttributeOutsideInit,PyUnusedLocal,PyProtectedMember
    def register_aggregator(self, transfer_variables: BaseTransferVariables, enable_secure_aggregate=True):
        self._enable_secure_aggregate = enable_secure_aggregate
        if enable_secure_aggregate:
            self._cipher = self._random_padding_cipher.create_cipher()
        return self

    def secure_aggregate(self, send_func, weights: Weights, degree: float = None, enable_secure_aggregate=True):
        # w -> w * degree
        if degree:
            weights *= degree
        # w * degree -> w * degree + \sum(\delta(i, j) * r_{ij}), namely， adding random mask.
        if enable_secure_aggregate:
            weights = weights.encrypted(cipher=self._cipher, inplace=True)
        # maybe remote degree
        remote_weights = weights.for_remote().with_degree(degree) if degree else weights.for_remote()

        send_func(remote_weights)

    def send_model(self, weights: Weights, degree: float = None, suffix=tuple()):
        def _func(_weights: TransferableWeights):
            self._model_scatter.send_model(model=_weights, suffix=suffix)

        return self.secure_aggregate(send_func=_func,
                                     weights=weights,
                                     degree=degree,
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


Guest = Client
Host = Client


def with_role(role, transfer_variable, enable_secure_aggregate=True):
    if role == consts.GUEST:
        return Client().register_aggregator(transfer_variable, enable_secure_aggregate)
    elif role == consts.HOST:
        return Client().register_aggregator(transfer_variable, enable_secure_aggregate)
    elif role == consts.ARBITER:
        return Arbiter().register_aggregator(transfer_variable, enable_secure_aggregate)
    else:
        raise ValueError(f"role {role} not found")
