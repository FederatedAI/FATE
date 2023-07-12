import typing

import numpy
from fate.arch import Context
from fate_utils.secure_aggregation_helper import DiffieHellman, MixAggregate, RandomMix


class _SecureAggregatorMeta:
    _send_name = "mixed_client_values"
    _recv_name = "aggregated_values"
    prefix: str

    def _get_name(self, name):
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name


class SecureAggregatorClient(_SecureAggregatorMeta):
    def __init__(self, prefix: typing.Optional[str] = None):
        self.prefix = prefix
        self._mixer = None

    def _get_mixer(self):
        if self._mixer is None:
            raise RuntimeError("mixer not initialized, run dh_exchange first")
        return self._mixer

    def dh_exchange(self, ctx: Context, ranks: typing.List[int]):
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
        mixed = self._get_mixer().mix(array, weight)
        print(mixed)
        ctx.arbiter.put(self._get_name(self._send_name), (mixed, weight))
        return ctx.arbiter.get(self._get_name(self._recv_name))


class SecureAggregatorServer(_SecureAggregatorMeta):
    def __init__(self, ranks, prefix: typing.Optional[str] = None):
        self.prefix = prefix
        self.ranks = ranks

    def secure_aggregate(self, ctx: Context):
        mix_aggregator = MixAggregate()
        aggregated_weight = 0.0
        has_weight = False
        for rank in self.ranks:
            mix_arrays, weight = ctx.parties[rank].get(self._get_name(self._send_name))
            mix_aggregator.aggregate(mix_arrays)
            if weight is not None:
                has_weight = True
                aggregated_weight += weight
        if not has_weight:
            aggregated_weight = None
        aggregated = mix_aggregator.finalize(aggregated_weight)
        for rank in self.ranks:
            ctx.parties[rank].put(self._get_name(self._recv_name), aggregated)
