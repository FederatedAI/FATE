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

from federatedml.frameworks.homo.coordinate.base import Coordinate
from federatedml.frameworks.homo.coordinate.transfer import arbiter_scatter
from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable


class SynchronizedPartyWeights(Coordinate):

    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        return SynchronizedPartyWeights(
            host_name=transfer_variable.host_party_weight.name,
            host_tag=transfer_variable.generate_transferid(transfer_variable.host_party_weight),
            guest_name=transfer_variable.guest_party_weight.name,
            guest_tag=transfer_variable.generate_transferid(transfer_variable.guest_party_weight)
        )

    def __init__(self, host_name, host_tag, guest_name, guest_tag):
        self._party_weight_scatter = arbiter_scatter(host_name=host_name,
                                                     host_tag=host_tag,
                                                     guest_name=guest_name,
                                                     guest_tag=guest_tag)

    def guest_call(self, party_weight):
        self._party_weight_scatter.remote_guest(party_weight)

    def host_call(self, party_weight):
        self._party_weight_scatter.remote_host(party_weight)

    def arbiter_call(self):
        weights = []
        guest_weight = self._party_weight_scatter.get_guest()
        hosts_weights = self._party_weight_scatter.remote_host()
        weights.append(guest_weight)
        weights.extend(hosts_weights)
        return [x / sum(weights) for x in weights]
