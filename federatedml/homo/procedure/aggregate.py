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

import operator
from functools import reduce

from federatedml.homo.sync import party_weights
from federatedml.homo.utils.scatter import scatter
from federatedml.homo.weights import Parameters
from federatedml.util import consts
from federatedml.util.transfer_variable.base_transfer_variable import Variable


class _Arbiter(object):
    def __init__(self,
                 host_party_weight_trv: Variable,
                 guest_party_weight_trv: Variable,
                 host_model_to_agg_trv: Variable,
                 guest_model_to_agg_trv: Variable):
        self._h_pw = host_party_weight_trv
        self._g_pw = guest_party_weight_trv
        self._h_model_to_agg = host_model_to_agg_trv
        self._g_model_to_agg = guest_model_to_agg_trv

        self._models = None
        self._party_weights = None

    def get_party_weights(self):
        self._party_weights = party_weights.arbiter(self._g_pw, self._h_pw).get_party_weights()
        return self._party_weights

    def _get_models(self, *suffix):
        self._models = scatter(self._h_model_to_agg, self._g_model_to_agg, suffix=suffix)

    def _decrypt_models(self, host_ciphers: dict):
        for i, cipher in host_ciphers.items():
            self._models[i + 1] = self._models[i].decrypted(cipher)

    def _mean_aggregate(self):
        num_clients = len(self._models)
        if not self._party_weights:
            agg_model = reduce(operator.add, self._models) / num_clients
        else:
            for m, w in zip(self._models, self._party_weights):
                m *= w
            agg_model = reduce(operator.add, self._models)
        self._models = None  # performed inplace operation, not correct anymore
        return agg_model

    def aggregate(self, version, cipher=None):
        self._get_models(version)
        if cipher:
            self._decrypt_models(cipher)
        return self._mean_aggregate()


class _Guest(object):
    def __init__(self,
                 guest_party_weight_trv: Variable,
                 guest_model_to_agg_trv: Variable):
        self._g_pw = guest_party_weight_trv
        self._g_model_to_agg = guest_model_to_agg_trv

    def send_party_weight(self, party_weight):
        party_weights.guest(self._g_pw).send(party_weight)

    def send_model(self, weights: Parameters, *suffix):
        self._g_model_to_agg.remote(obj=weights.for_remote(), role=consts.ARBITER, idx=0, suffix=suffix)


class _Host(object):
    def __init__(self,
                 host_party_weight_trv: Variable,
                 host_model_to_agg_trv: Variable):
        self._h_pw = host_party_weight_trv
        self._h_model_to_agg = host_model_to_agg_trv

    def send_party_weight(self, party_weight):
        party_weights.guest(self._h_pw).send(party_weight)

    def send_model(self, weights: Parameters, *suffix):
        self._h_model_to_agg.remote(obj=weights.for_remote(), role=consts.ARBITER, idx=0, suffix=suffix)


def arbiter(host_party_weight_trv: Variable,
            guest_party_weight_trv: Variable,
            host_model_to_agg_trv: Variable,
            guest_model_to_agg_trv: Variable):
    return _Arbiter(host_party_weight_trv, guest_party_weight_trv, host_model_to_agg_trv, guest_model_to_agg_trv)


def guest(guest_party_weight_trv: Variable, guest_model_to_agg_trv: Variable):
    return _Guest(guest_party_weight_trv, guest_model_to_agg_trv)


def host(host_party_weight_trv: Variable, host_model_to_agg_trv: Variable):
    return _Host(host_party_weight_trv, host_model_to_agg_trv)
