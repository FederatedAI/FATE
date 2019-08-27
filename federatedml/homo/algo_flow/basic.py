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

from federatedml.homo.procedure import aggregated_bc
from federatedml.homo.procedure import aggregator
from federatedml.util.transfer_variable.base_transfer_variable import Variable


class Arbiter(object):

    def __init__(self, ):
        self.random_padding_cipher = None
        self.aggregator = None
        self.aggregated_bc = None
        self.model = None
        self.ciphers = None

    def register_aggregator(self,
                            host_party_weight_trv: Variable,
                            guest_party_weight_trv: Variable,
                            host_model_to_agg_trv: Variable,
                            guest_model_to_agg_trv: Variable):
        self.aggregator = aggregator.arbiter(
            host_party_weight_trv,
            guest_party_weight_trv,
            host_model_to_agg_trv,
            guest_model_to_agg_trv)

    def register_aggregated_bc(self, aggregated_model_trv: Variable):
        self.aggregated_bc = aggregated_bc.arbiter(aggregated_model_trv)

    def initialize(self):
        self.aggregator.get_party_weights()

    def aggregate(self, suffix=tuple(), arbiter_trainer=None):
        """
        aggregate models, then send aggregated model to guest and host.

        if `arbiter_trainer` provided, aggregated model would be processed before send.
        Args:
            suffix:
            arbiter_trainer:
        """
        self.model = self.aggregator.aggregate(suffix=suffix, cipher=self.ciphers)
        if arbiter_trainer:
            self.model = arbiter_trainer(self.model)
        self.aggregated_bc.send(self.model, ciphers=self.ciphers, suffix=suffix)


class Guest(object):

    def __init__(self):
        self.aggregator = None
        self.aggregated_bc = None

    def register_aggregator(self, guest_party_weight_trv: Variable, guest_model_to_agg_trv: Variable):
        self.aggregator = aggregator.guest(guest_party_weight_trv, guest_model_to_agg_trv)

    def register_aggregated_bc(self, aggregated_model_trv: Variable):
        self.aggregated_bc = aggregated_bc.guest(aggregated_model_trv)

    def initialize(self, party_weight):
        self.aggregator.send_party_weight(party_weight)

    def aggregate(self, model, suffix):
        self.aggregator.send_model(weights=model, suffix=suffix)
        return self.aggregated_bc.get(suffix=suffix)


class Host(object):

    def __init__(self):
        self.aggregator = None
        self.aggregated_bc = None

    def register_aggregator(self, host_party_weight_trv: Variable, host_model_to_agg_trv: Variable):
        self.aggregator = aggregator.host(host_party_weight_trv, host_model_to_agg_trv)

    def register_aggregated_bc(self, aggregated_model_trv: Variable):
        self.aggregated_bc = aggregated_bc.host(aggregated_model_trv)

    def initialize(self, party_weight):
        self.aggregator.send_party_weight(party_weight)

    def aggregate(self, model, suffix):
        self.aggregator.send_model(weights=model, suffix=suffix)
        return self.aggregated_bc.get(suffix=suffix)


def arbiter():
    return Arbiter()


def guest():
    return Guest()


def host():
    return Host()
