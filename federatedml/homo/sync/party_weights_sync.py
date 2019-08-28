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

from federatedml.homo.utils.scatter import Scatter
from federatedml.util import consts


# noinspection PyAttributeOutsideInit
class Arbiter(object):

    def _register_party_weights_transfer(self, guest_party_weight_transfer, host_party_weight_transfer):
        self._scatter = Scatter(guest_party_weight_transfer, host_party_weight_transfer)

    def get_party_weights(self):
        weights = list(self._scatter.get())
        total = sum(weights)
        self._party_weights = [x / total for x in weights]
        return self._party_weights


class _Client(object):

    # noinspection PyAttributeOutsideInit
    def _register_party_weights_transfer(self, transfer_variable):
        self._transfer_variable = transfer_variable

    def send_party_weight(self, obj):
        self._transfer_variable.remote(obj=obj, role=consts.ARBITER, idx=0)


Host = _Client
Guest = _Client
