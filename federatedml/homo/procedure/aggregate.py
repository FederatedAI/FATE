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

from federatedml.homo.weights import Variables
from federatedml.homo.sync.party_weights import PartyWeightsProcedures
from federatedml.homo.sync.scatter_parameters import ScatterParameters


def _tag_suffix(version):
    return f"epoch_{version}"


class _Arbiter(object):
    def __init__(self, transfer_variable):
        self._host_models = None
        self._guest_model = None
        self._transfer_variable = transfer_variable
        self._scatter = ScatterParameters.arbiter(transfer_variable)
        self._party_weights = None

    def get_party_weights(self):
        self._party_weights = PartyWeightsProcedures.arbiter(self._transfer_variable).get_party_weights()
        return self._party_weights

    def get_models(self, version):
        # receive host models

        self._host_models = [Variables.from_transferable(v)
                             for v in self._scatter.get_hosts(suffix=_tag_suffix(version))]

        # receive guest models
        self._guest_model = Variables.from_transferable(self._scatter.get_guest(suffix=_tag_suffix(version)))

    def decrypt_models(self, ciphers: dict):
        # decrypt model by paillier ciphers
        for i, cipher in ciphers.items():
            self._host_models[i] = self._host_models[i].decrypted(cipher)

    def mean_aggregate(self):
        num_clients = len(self._host_models) + 1
        if not self._party_weights:
            agg_model = self._guest_model
            for model in self._host_models:
                agg_model += model
                agg_model /= float(num_clients)
        else:
            agg_model = self._guest_model
            agg_model *= self._party_weights[0]

            for i, model in enumerate(self._host_models):
                agg_model.axpy(self._party_weights[i + 1], model)

        return agg_model

    def aggregate(self, version, cipher=None):
        self.get_models(version)
        if cipher:
            self.decrypt_models(cipher)
        return self.mean_aggregate()


class _Guest(object):
    def __init__(self, transfer_variable):
        self._transfer_variable = transfer_variable
        self._scatter = ScatterParameters.guest(transfer_variable)

    def send_party_weight(self, party_weight):
        PartyWeightsProcedures.guest(self._transfer_variable).remote_party_weight(party_weight)

    def send(self, weights: Variables, version=0):
        self._scatter.remote_guest(weights.for_remote(), suffix=_tag_suffix(version))


class _Host(object):
    def __init__(self, transfer_variable):
        self._transfer_variable = transfer_variable
        self._scatter = ScatterParameters.host(transfer_variable)

    def send_party_weight(self, party_weight):
        PartyWeightsProcedures.host(self._transfer_variable).remote_party_weight(party_weight)

    def send(self, weights: Variables, version):
        self._scatter.remote_host(weights.for_remote(), suffix=_tag_suffix(version))


class Aggregate(object):
    """@hosts, @guest -> @arbiter
    transfer models from hosts and guest to arbiter for model aggregation
    """

    @staticmethod
    def arbiter(transfer_variable):
        return _Arbiter(transfer_variable)

    @staticmethod
    def guest(transfer_variable):
        return _Guest(transfer_variable)

    @staticmethod
    def host(transfer_variable):
        return _Host(transfer_variable)
