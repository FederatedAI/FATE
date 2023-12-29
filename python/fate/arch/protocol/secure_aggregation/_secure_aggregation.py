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

import typing

import numpy
from fate.arch import Context
from fate.arch.protocol.diffie_hellman import DiffieHellman
from fate_utils.secure_aggregation_helper import MixAggregate, RandomMix


class _SecureAggregatorMeta:
    _send_name = "mixed_client_values"
    _recv_name = "aggregated_values"
    prefix: str

    def _get_name(self, name):
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name


class SecureAggregatorClient(_SecureAggregatorMeta):
    def __init__(self, prefix: typing.Optional[str] = None, is_mock: bool = False):
        """
        secure aggregation client
        Args:
            prefix: unique prefix for this aggregator
            is_mock: mock the aggregator, do not perform secure aggregation, for test only
        """
        self.prefix = prefix
        self._mixer = None
        self._is_mock = is_mock

    def _get_mixer(self):
        if self._mixer is None:
            raise RuntimeError("mixer not initialized, run dh_exchange first")
        return self._mixer

    def dh_exchange(self, ctx: Context, ranks: typing.List[int]):
        if self._is_mock:
            return
        local_rank = ctx.local.rank
        dh = {}
        seeds = {}
        for rank in ranks:
            if rank == local_rank:
                continue
            dh[rank] = DiffieHellman()
            ctx.parties[rank].put(self._get_name(f"dh_pubkey"), dh[rank].get_public_key())
        for rank in ranks:
            if rank == local_rank:
                continue
            public_key = ctx.parties[rank].get(self._get_name(f"dh_pubkey"))
            seeds[rank] = dh[rank].diffie_hellman(public_key)
        self._mixer = RandomMix(seeds, local_rank)

    def secure_aggregate(self, ctx: Context, array: typing.List[numpy.ndarray], weight: typing.Optional[int] = None):
        if self._is_mock:
            ctx.arbiter.put(self._get_name(self._send_name), (array, weight))
            return ctx.arbiter.get(self._get_name(self._recv_name))
        else:
            mixed = self._get_mixer().mix(array, weight)
            ctx.arbiter.put(self._get_name(self._send_name), (mixed, weight))
            return ctx.arbiter.get(self._get_name(self._recv_name))


class SecureAggregatorServer(_SecureAggregatorMeta):
    def __init__(self, ranks, prefix: typing.Optional[str] = None, is_mock: bool = False):
        """
        secure aggregation server
        Args:
            ranks: all ranks
            prefix: unique prefix for this aggregator
            is_mock: mock the aggregator, do not perform secure aggregation, for test only
        """
        self.prefix = prefix
        self.ranks = ranks
        self._is_mock = is_mock

    def secure_aggregate(self, ctx: Context, ranks: typing.Optional[int] = None):
        """
        perform secure aggregate once
        Args:
            ctx: Context to use
            ranks: ranks to aggregate, if None, use all ranks
        """
        if ranks is None:
            ranks = self.ranks
        aggregated_weight = 0.0
        has_weight = False

        if self._is_mock:
            aggregated = []
            for rank in ranks:
                arrays, weight = ctx.parties[rank].get(self._get_name(self._send_name))
                for i in range(len(arrays)):
                    if len(aggregated) <= i:
                        aggregated.append(arrays[i])
                    else:
                        aggregated[i] += arrays[i]
                if weight is not None:
                    has_weight = True
                    aggregated_weight += weight
            if has_weight:
                aggregated = [x / aggregated_weight for x in aggregated]
        else:
            mix_aggregator = MixAggregate()
            for rank in ranks:
                mix_arrays, weight = ctx.parties[rank].get(self._get_name(self._send_name))
                mix_aggregator.aggregate(mix_arrays)
                if weight is not None:
                    has_weight = True
                    aggregated_weight += weight
            if not has_weight:
                aggregated_weight = None
            aggregated = mix_aggregator.finalize(aggregated_weight)

        for rank in ranks:
            ctx.parties[rank].put(self._get_name(self._recv_name), aggregated)

        return aggregated
