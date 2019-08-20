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
from federatedml.frameworks.homo.coordinate.base import Coordinate
from federatedml.frameworks.homo.coordinate.transfer import arbiter_scatter
from federatedml.util import consts
from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable


class Aggregate(Coordinate):
    """@hosts, @guest -> @arbiter
    transfer models from hosts and guest to arbiter for model aggregation
    """

    def __init__(self,
                 aggregate_host_name,
                 aggregate_host_tag,
                 aggregate_guest_name,
                 aggregate_guest_tag):
        self._model_scatter = arbiter_scatter(host_name=aggregate_host_name,
                                              host_tag=aggregate_host_tag,
                                              guest_name=aggregate_guest_name,
                                              guest_tag=aggregate_guest_tag)

    def guest_call(self, weights: TransferableWeights, tag_suffix):
        self._model_scatter.remote_guest(weights, suffix=tag_suffix)

    def host_call(self, weights: TransferableWeights, tag_suffix):
        return self._model_scatter.remote_guest(weights, suffix=tag_suffix)

    def arbiter_call(self, party_weights, epoch, paillier_ciphers=None, tag_suffix=None):
        # receive host models
        host_models = self._model_scatter.get_hosts(suffix=tag_suffix)

        # decrypt model by paillier ciphers
        if paillier_ciphers:
            for i in range(len(paillier_ciphers)):
                host_models[i] = paillier_ciphers[i].decript(host_models[i])

        # receive guest models
        guest_model = self._model_scatter.get_guest(suffix=tag_suffix)

        if not party_weights:
            party_weights = [1.0 / (len(host_models) + 1)] * (len(host_models) + 1)

        final_model = guest_model

        for k, v in final_model.items():
            final_model[k] = final_model[k] * party_weights[0]

        for k, v in final_model.items():
            for i, model in enumerate(host_models):
                final_model[k] += model[k] * party_weights[i+1]

        return final_model


class ModelAggregate(Aggregate):

    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        return Aggregate(
            aggregate_host_name=transfer_variable.host_model.name,
            aggregate_host_tag=transfer_variable.generate_transferid(transfer_variable.host_model),
            aggregate_guest_name=transfer_variable.guest_model.name,
            aggregate_guest_tag=transfer_variable.generate_transferid(transfer_variable.guest_model)
        )


class GradientAggregate(Aggregate):
    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        return Aggregate(
            aggregate_host_name=transfer_variable.host_gradient.name,
            aggregate_host_tag=transfer_variable.generate_transferid(transfer_variable.host_gradient),
            aggregate_guest_name=transfer_variable.guest_gradient.name,
            aggregate_guest_tag=transfer_variable.generate_transferid(transfer_variable.guest_gradient)
        )
