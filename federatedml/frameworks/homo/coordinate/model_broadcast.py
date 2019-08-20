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

from federatedml.frameworks.homo.model.weights import TransferableWeights
from federatedml.frameworks.homo.coordinate.transfer import arbiter_broadcast
from federatedml.frameworks.homo.coordinate.base import Coordinate
from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable


class Broadcast(Coordinate):
    """@Arbiter -> [@Host, @Guest]
    transfer model from arbiter to hosts and guest
    """

    def __init__(self, broadcast_name, broadcast_tag):
        self._model_broadcast = arbiter_broadcast(name=broadcast_name,
                                                  tag=broadcast_tag)

    def guest_call(self, tag_suffix):
        return self._model_broadcast.get(suffix=tag_suffix)

    def host_call(self, tag_suffix):
        return self._model_broadcast.get(suffix=tag_suffix)

    def arbiter_call(self, model_weights: TransferableWeights, tag_suffix, paillier_ciphers=None):
        if paillier_ciphers:
            for idx, cipher in enumerate(paillier_ciphers):
                encrypt_model = model_weights.encrypted(cipher, inplace=False)
                self._model_broadcast.remote(encrypt_model, suffix=tag_suffix, idx=idx)
        else:
            self._model_broadcast.remote(model_weights, suffix=tag_suffix, idx=-1)


class GradientBroadcast(Broadcast):
    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        return Broadcast(broadcast_name=transfer_variable.mean_gradient.name,
                         broadcast_tag=transfer_variable.generate_transferid(transfer_variable.mean_gradient))


class ModelBroadcast(Broadcast):
    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        return Broadcast(broadcast_name=transfer_variable.mean_model.name,
                         broadcast_tag=transfer_variable.generate_transferid(transfer_variable.mean_model))