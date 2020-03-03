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

from arch.api.utils import log_utils
from federatedml.framework.homo.blocks import aggregator
from federatedml.framework.homo.blocks import random_padding_cipher
from federatedml.framework.homo.blocks.aggregator import AggregatorTransVar
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.random_padding_cipher import RandomPaddingCipherTransVar
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class SecureAggregatorTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.aggregator_trans_var = AggregatorTransVar(server=server, clients=clients, prefix=self.prefix)
        self.random_padding_cipher_trans_var = \
            RandomPaddingCipherTransVar(server=server, clients=clients, prefix=self.prefix)


class Server(object):
    def __init__(self, trans_var: SecureAggregatorTransVar = SecureAggregatorTransVar(), enable_secure_aggregate=True):
        self._aggregator = aggregator.Server(trans_var=trans_var.aggregator_trans_var)
        self.enable_secure_aggregate = enable_secure_aggregate
        if enable_secure_aggregate:
            random_padding_cipher.Server(trans_var=trans_var.random_padding_cipher_trans_var) \
                .exchange_secret_keys()

    def aggregate_model(self, suffix=tuple()):
        models = self._aggregator.get_models(suffix=suffix)
        total_model, total_degree = models[0]
        for model, degree in models[1:]:
            total_model = model_add(total_model, model)
            total_degree += degree
        mean_model = model_div_scalar(total_model, total_degree)
        return mean_model

    def send_aggregated_model(self, model, suffix=tuple()):
        self._aggregator.send_aggregated_model(model=model, suffix=suffix)


class Client(object):
    def __init__(self, trans_var: SecureAggregatorTransVar = SecureAggregatorTransVar(), enable_secure_aggregate=True):
        self.enable_secure_aggregate = enable_secure_aggregate
        self._aggregator = aggregator.Client(trans_var=trans_var.aggregator_trans_var)
        if enable_secure_aggregate:
            self._random_padding_cipher = \
                random_padding_cipher.Client(trans_var=trans_var.random_padding_cipher_trans_var).create_cipher()

    def send_model(self, model, degree: float = None, suffix=tuple()):
        # w -> w * degree
        if degree is not None:
            model = model_mul_scalar(model, degree)
        else:
            degree = 1.0
        # w * degree -> w * degree + \sum(\delta(i, j) * r_{ij}), namely， adding random mask.
        if self.enable_secure_aggregate:
            model = model_encrypted(model, cipher=self._random_padding_cipher)

        self._aggregator.send_model((model, degree), suffix=suffix)

    def get_aggregated_model(self, suffix=tuple()):
        return self._aggregator.get_aggregated_model(suffix=suffix)


@functools.singledispatch
def model_encrypted(model, cipher):
    return model.encrypted(cipher, inplace=True)


@functools.singledispatch
def model_add(model, other):
    return model + other


@functools.singledispatch
def model_mul_scalar(model, scalar):
    model *= scalar
    return model


@functools.singledispatch
def model_div_scalar(model, scalar):
    model /= scalar
    return model
