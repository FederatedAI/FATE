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
from federatedml.framework.homo.blocks import secure_aggregator
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar, model_cipher_func
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class SecureMeanAggregatorTransVar(SecureAggregatorTransVar):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)


class Server(secure_aggregator.Server):
    def __init__(self, trans_var: SecureAggregatorTransVar = SecureMeanAggregatorTransVar(),
                 enable_secure_aggregate=True):
        super().__init__(trans_var=trans_var, enable_secure_aggregate=enable_secure_aggregate)

    def mean_model(self, suffix=tuple()):
        def _func(models):
            num = len(models)
            return model_div_scalar(functools.reduce(model_add, models), float(num))

        return self.aggregate(_func, suffix=suffix)

    def weighted_mean_model(self, suffix=tuple()):
        def _func(models):
            total_model, total_degree = models[0]
            for model, degree in models[1:]:
                total_model = model_add(total_model, model)
                total_degree += degree
            mean_model = model_div_scalar(total_model, total_degree)
            return mean_model

        return self.aggregate(_func, suffix=suffix)


class Client(secure_aggregator.Client):
    def __init__(self, trans_var: SecureAggregatorTransVar = SecureAggregatorTransVar(), enable_secure_aggregate=True):
        super().__init__(trans_var=trans_var, enable_secure_aggregate=enable_secure_aggregate)

    def send_weighted_model(self, weighted_model, weight: float, suffix=tuple()):
        # w -> w * weight
        weighted_model = model_mul_scalar(weighted_model, weight)
        # w * weight -> w * weight + \sum(\delta(i, j) * r_{ij}), namely， adding random mask.
        if self.enable_secure_aggregate:
            model_cipher = model_cipher_func(weighted_model)
            weighted_model = model_cipher(self._random_padding_cipher)

        self._aggregator.send_model((weighted_model, weight), suffix=suffix)


@functools.singledispatch
def model_add(model, other):
    return model + other


@functools.singledispatch
def model_mul_scalar(model, other):
    return model * other


@functools.singledispatch
def model_div_scalar(model, scalar):
    model /= scalar
    return model
