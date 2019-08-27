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

from federatedml.homo.sync import party_weights_sync, model_transfer_sync, loss_transfer_sync


class Arbiter(party_weights_sync.Arbiter, model_transfer_sync.Arbiter, loss_transfer_sync.Arbiter):

    # noinspection PyAttributeOutsideInit
    def initialize_aggregator(self, use_party_weight=True):
        if use_party_weight:
            self._party_weights = self.get_party_weights()

    def register_aggregator(self, transfer_variables):
        """
        register transfer of party_weights, models and losses.
        Args:
            transfer_variables: assuming transfer_variable has variables:
                1. guest_party_weight,  host_party_weight for party_weights transfer
                2. guest_model_transfer, host_model_transfer for model transfer
                3. guest_loss_transfer, host_loss_transfer for loss transfer
        """
        self.register_party_weights_transfer(guest_party_weight_transfer=transfer_variables.guest_party_weight,
                                             host_party_weight_transfer=transfer_variables.host_party_weight)

        self.register_model_transfer(host_model_transfer=transfer_variables.host_model_transfer,
                                     guest_model_transfer=transfer_variables.guest_model_transfer)

        self.register_loss_transfer(host_loss_transfer=transfer_variables.host_loss_transfer,
                                    guest_loss_transfer=transfer_variables.guest_loss_transfer)

    def aggregate_model(self, ciphers_dict=None, suffix=tuple()):
        models = self.get_models(ciphers_dict, suffix=suffix)
        num_clients = len(models)
        if not self._party_weights:
            return reduce(operator.add, models) / num_clients
        for m, w in zip(models, self._party_weights):
            m *= w
        return reduce(operator.add, models)

    def aggregate_loss(self, idx=None, suffix=tuple()):
        losses = self.get_losses(idx=idx, suffix=suffix)
        if idx is None:
            return sum(map(lambda pair: pair[0] * pair[1], zip(losses, self._party_weights)))
        else:
            total_weights = self._party_weights[0]
            loss = losses[0]
            for party_id in idx:
                total_weights += self._party_weights[party_id]
                loss += losses[party_id]
            return loss / total_weights


class Guest(party_weights_sync.Guest, model_transfer_sync.Guest, loss_transfer_sync.Guest):

    def initialize_aggregator(self, party_weight):
        self.send_party_weight(party_weight)

    def register_aggregator(self, transfer_variables):
        """
           register transfer of party_weights, models and losses.
           Args:
               transfer_variables: assuming transfer_variable has variables:
                   1. guest_party_weight for party_weights transfer
                   2. guest_model_transfer for model transfer
                   3. guest_loss_transfer for loss transfer
        """
        self.register_party_weights_transfer(transfer_variable=transfer_variables.guest_party_weight)

        self.register_model_transfer(model_transfer=transfer_variables.guest_model_transfer)

        self.register_loss_transfer(loss_transfer=transfer_variables.guest_loss_transfer)


class Host(party_weights_sync.Host, model_transfer_sync.Host, loss_transfer_sync.Host):

    def initialize_aggregator(self, party_weight):
        self.send_party_weight(party_weight)

    def register_aggregator(self, transfer_variables):
        """
           register transfer of party_weights, models and losses.
           Args:
               transfer_variables: assuming transfer_variable has variables:
                   1. host_party_weight for party_weights transfer
                   2. host_model_transfer for model transfer
                   3. host_loss_transfer for loss transfer
        """
        self.register_party_weights_transfer(transfer_variable=transfer_variables.host_party_weight)

        self.register_model_transfer(model_transfer=transfer_variables.host_model_transfer)

        self.register_loss_transfer(loss_transfer=transfer_variables.host_loss_transfer)
