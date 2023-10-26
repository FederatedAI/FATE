import logging
import typing

from .communicator_base import Communicator as CommunicatorBase
from fate.arch.context import Context
from fate.arch.context._namespace import NS
import torch
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from fate.arch.context._federation_mpc import WrapCommunicator


class Communicator(CommunicatorBase):
    """
    FATECommunicator is a wrapper around the FATE communicator.
    """

    instance = None

    def __init__(self, federation: "WrapCommunicator", rank, world_size):
        self.federation = federation
        self.rank = rank
        self.world_size = world_size

        self.main_group = None

    @classmethod
    def is_initialized(cls):
        return cls.instance is not None

    @classmethod
    def get(cls):
        return cls.instance

    def _assert_initialized(self):
        assert self.is_initialized(), "initialize the communicator first"

    def get_rank(self):
        self._assert_initialized()
        return self.rank

    def get_world_size(self):
        self._assert_initialized()
        return self.world_size

    @classmethod
    def initialize(cls, ctx: Context, init_ttp):
        from fate.arch.context._federation_mpc import WrapCommunicator

        # environment variables first
        world_size = ctx.world_size
        rank = ctx.local.rank
        namespace = NS(ctx.namespace.sub_ns("mpc"), 0)
        rank_to_party = {p.rank: p.party for p in ctx.parties}
        cls.instance = Communicator(
            federation=WrapCommunicator(
                ctx,
                namespace,
                rank_to_party,
                rank,
                world_size,
            ),
            rank=rank,
            world_size=world_size,
        )

    @classmethod
    def shutdown(cls):
        pass

    def send(self, tensor, dst):
        self.federation.send(tensor, dst)

    def recv(self, tensor, src=None):
        self.federation.recv(tensor, src)

    def isend(self, tensor, dst):
        return self.federation.isend(tensor, dst)

    def irecv(self, tensor: torch.Tensor, src=None):
        return self.federation.irecv(tensor, src)

    def scatter(self, scatter_list, src, size=None, async_op=False):
        raise NotImplementedError

    def reduce(self, tensor, op=None, async_op=False):
        raise NotImplementedError

    def all_reduce(self, input, op=ReduceOp.SUM, batched=False):
        if batched:
            assert isinstance(input, list), "batched reduce input must be a list"
            results = []
            for tensor in input:
                results.append(self.all_reduce(tensor, op, batched=False))
            return results
        else:
            return self.federation.all_reduce(input, op)

    def gather(self, tensor, dst, async_op=False):
        raise NotImplementedError

    def all_gather(self, tensor, async_op=False):
        return self.federation.all_gather(tensor, async_op)

    def broadcast(self, input, src, group=None, batched=False):
        self._assert_initialized()
        group = self.main_group if group is None else group
        if batched:
            assert isinstance(input, list), "batched reduce input must be a list"
            reqs = []
            for tensor in input:
                reqs.append(self.federation.broadcast(tensor.data, src, group=group, async_op=True))
            for req in reqs:
                req.wait()
        else:
            assert torch.is_tensor(input.data), "unbatched input for reduce must be a torch tensor"
            self.federation.broadcast(input.data, src, group=group)
        return input
