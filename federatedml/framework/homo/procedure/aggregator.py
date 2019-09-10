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

from federatedml.framework.homo.sync import party_weights_sync, model_scatter_sync, \
    loss_transfer_sync, model_broadcast_sync, is_converge_sync
from federatedml.framework.weights import Weights


class Arbiter(party_weights_sync.Arbiter,
              model_scatter_sync.Arbiter,
              model_broadcast_sync.Arbiter,
              loss_transfer_sync.Arbiter,
              is_converge_sync.Arbiter):

    # noinspection PyAttributeOutsideInit
    def initialize_aggregator(self, use_party_weight=True):
        if use_party_weight:
            self._party_weights = self.get_party_weights()

    def register_aggregator(self, transfer_variables):
        """
        register transfer of party_weights, models and losses.
        Args:
            transfer_variables: assuming transfer_variable has variables:
                1. guest_party_weight,  host_party_weight for party_weights scatter
                2. guest_model, host_model for model scatter
                3. aggregated_model for broadcast aggregated model
                4. guest_loss, host_loss for loss scatter
        """
        self._register_party_weights_transfer(guest_party_weight_transfer=transfer_variables.guest_party_weight,
                                              host_party_weight_transfer=transfer_variables.host_party_weight)

        self._register_model_scatter(host_model_transfer=transfer_variables.host_model,
                                     guest_model_transfer=transfer_variables.guest_model)
        self._register_model_broadcaster(model_transfer=transfer_variables.aggregated_model)

        self._register_loss_transfer(host_loss_transfer=transfer_variables.host_loss,
                                     guest_loss_transfer=transfer_variables.guest_loss)

        self._register_is_converge(is_converge_variable=transfer_variables.is_converge)

    def aggregate_model(self, ciphers_dict=None, suffix=tuple()) -> Weights:
        models = self.get_models_for_aggregate(ciphers_dict, suffix=suffix)
        num_clients = len(models)
        if not self._party_weights:
            return reduce(operator.add, models) / num_clients
        for m, w in zip(models, self._party_weights):
            m *= w
        return reduce(operator.add, models)

    def aggregate_and_broadcast(self, ciphers_dict=None, suffix=tuple()):
        """
        aggregate models from guest and hosts, then broadcast the aggregated model.

        Args:
            ciphers_dict: a dict of host id to host cipher
            suffix: tag suffix
        """
        model = self.aggregate_model(ciphers_dict=ciphers_dict, suffix=suffix)
        self._send_model(model, ciphers_dict=ciphers_dict, suffix=suffix)
        return model

    def get_models_for_aggregate(self, ciphers_dict=None, suffix=tuple()):
        return self._get_models(ciphers_dict=ciphers_dict, suffix=suffix)

    def send_aggregated_model(self, model: Weights, ciphers_dict=None, suffix=tuple()):
        self._send_model(model=model, ciphers_dict=ciphers_dict, suffix=suffix)

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


class Guest(party_weights_sync.Guest,
            model_scatter_sync.Guest,
            loss_transfer_sync.Guest,
            model_broadcast_sync.Guest,
            is_converge_sync.Guest):

    def initialize_aggregator(self, party_weight):
        self.send_party_weight(party_weight)

    def register_aggregator(self, transfer_variables):
        """
           register transfer of party_weights, models and losses.
           Args:
               transfer_variables: assuming transfer_variable has variables:
                   1. guest_party_weight to send party_weights
                   2. guest_model to send model for aggregate
                   3. aggregated_model to get aggregated model
                   4. guest_loss for loss send
        """
        self._register_party_weights_transfer(transfer_variable=transfer_variables.guest_party_weight)

        self._register_model_scatter(model_transfer=transfer_variables.guest_model)

        self._register_model_broadcaster(model_transfer=transfer_variables.aggregated_model)

        self._register_loss_transfer(loss_transfer=transfer_variables.guest_loss)

        self._register_is_converge(is_converge_variable=transfer_variables.is_converge)

    def aggregate_and_get(self, model: Weights, suffix=tuple()):
        self.send_model_for_aggregate(weights=model, suffix=suffix)
        return self.get_aggregated_model(suffix=suffix)

    def get_aggregated_model(self, suffix=tuple()):
        return self._get_model(suffix=suffix)

    def send_model_for_aggregate(self, weights: Weights, suffix=tuple()):
        self._send_model(weights=weights, suffix=suffix)


class Host(party_weights_sync.Host,
           model_scatter_sync.Host,
           loss_transfer_sync.Host,
           model_broadcast_sync.Host,
           is_converge_sync.Host):

    def initialize_aggregator(self, party_weight):
        self.send_party_weight(party_weight)

    def register_aggregator(self, transfer_variables):
        """
           register transfer of party_weights, models and losses.
           Args:
               transfer_variables: assuming transfer_variable has variables:
                    1. host_party_weight to send party_weights
                    2. host_model to send model for aggregate
                    3. aggregated_model to get aggregated model
                    4. host_loss for loss send
        """
        self._register_party_weights_transfer(transfer_variable=transfer_variables.host_party_weight)

        self._register_model_scatter(model_transfer=transfer_variables.host_model)

        self._register_model_broadcaster(model_transfer=transfer_variables.aggregated_model)

        self._register_loss_transfer(loss_transfer=transfer_variables.host_loss)

        self._register_is_converge(is_converge_variable=transfer_variables.is_converge)

    def aggregate_and_get(self, model: Weights, suffix=tuple()):
        self.send_model_for_aggregate(weights=model, suffix=suffix)
        return self.get_aggregated_model(suffix=suffix)

    def send_model_for_aggregate(self, weights: Weights, suffix=tuple()):
        self._send_model(weights=weights, suffix=suffix)

    def get_aggregated_model(self, suffix=tuple()):
        return self._get_model(suffix=suffix)
