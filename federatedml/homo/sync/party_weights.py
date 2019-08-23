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

from federatedml.homo.transfer import Scatter
from federatedml.homo.transfer import arbiter_scatter


class _Arbiter(object):
    def __init__(self, party_weights_scatter: Scatter):
        self._party_weights_scatter = party_weights_scatter

    def get_party_weights(self):
        weights = []
        guest_weight = self._party_weights_scatter.get_guest()
        hosts_weights = self._party_weights_scatter.get_hosts()
        weights.append(guest_weight)
        weights.extend(hosts_weights)
        return [x / sum(weights) for x in weights]


class _Guest(object):
    def __init__(self, party_weights_scatter: Scatter):
        self._party_weights_scatter = party_weights_scatter

    def remote_party_weight(self, party_weight):
        self._party_weights_scatter.remote_guest(party_weight)


class _Host(object):
    def __init__(self, party_weights_scatter: Scatter):
        self._party_weights_scatter = party_weights_scatter

    def remote_party_weight(self, party_weight):
        self._party_weights_scatter.remote_host(party_weight)


def _parse_transfer_variable(transfer_variable):
    return arbiter_scatter(host_name=transfer_variable.host_party_weight.name,
                           host_tag=transfer_variable.generate_transferid(transfer_variable.host_party_weight),
                           guest_name=transfer_variable.guest_party_weight.name,
                           guest_tag=transfer_variable.generate_transferid(transfer_variable.guest_party_weight))


class PartyWeightsProcedures(object):

    @staticmethod
    def arbiter(transfer_variable):
        return _Arbiter(_parse_transfer_variable(transfer_variable))

    @staticmethod
    def guest(transfer_variable):
        return _Guest(_parse_transfer_variable(transfer_variable))

    @staticmethod
    def host(transfer_variable):
        return _Host(_parse_transfer_variable(transfer_variable))
