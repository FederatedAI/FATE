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

from federatedml.homo.weights import Parameters
from federatedml.util.transfer_variable.base_transfer_variable import Variable
from federatedml.util import consts


class _Arbiter(object):
    def __init__(self, aggregated_model_trv: Variable):
        self._aggregated_model_trv = aggregated_model_trv

    def send(self, model_weights: Parameters, ciphers, suffix):
        if ciphers:
            for idx, cipher in ciphers.items():
                encrypt_model = model_weights.encrypted(cipher, inplace=False)
                self._aggregated_model_trv.remote(obj=encrypt_model.for_remote(),
                                                  role=consts.HOST,
                                                  idx=idx,
                                                  suffix=suffix)
            self._aggregated_model_trv.remote(obj=model_weights.for_remote(),
                                              role=consts.GUEST,
                                              idx=0,
                                              suffix=suffix)
        else:
            self._aggregated_model_trv.remote(obj=model_weights.for_remote(),
                                              role=None,
                                              idx=-1,
                                              suffix=suffix)


class _Client(object):
    def __init__(self, aggregated_model_trv: Variable):
        self._aggregated_model_trv = aggregated_model_trv

    def get(self, suffix):
        return self._aggregated_model_trv.get(idx=0, suffix=suffix)


def arbiter(aggregated_model_trv: Variable):
    return _Arbiter(aggregated_model_trv)


def host(aggregated_model_trv: Variable):
    return _Client(aggregated_model_trv)


def guest(aggregated_model_trv):
    return _Client(aggregated_model_trv)
