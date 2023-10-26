import functools
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed import ReduceOp

from ._namespace import NS
from ._context import Context
from ._federation import Parties

logger = logging.getLogger(__name__)


class MPCFederation:
    def __init__(self, rank_to_party, rank, world_size, namespace, ctx: Context):
        self.ctx = ctx
        self.rank_to_party = rank_to_party
        self.rank = rank
        self.world_size = world_size
        self.namespace = namespace

    def _get_parties(self, parties):
        return Parties(
            self.ctx,
            self.ctx.federation,
            [(i, p) for i, p in enumerate(parties)],
            self.namespace,
        )

    def _get_parties_by_rank(self, rank):
        return self._get_parties([self.rank_to_party[rank]])

    def _get_parties_by_ranks(self, ranks):
        return self._get_parties([self.rank_to_party[rank] for rank in ranks])

    def send(self, index, tensor, dst):
        parties = self._get_parties_by_rank(dst)
        logger.debug(f"[{self.ctx.local}]sending, index={index}, dst={dst}, parties={parties}")
        parties.put(self.namespace.indexed_ns(index).federation_tag, tensor)

    def recv(self, index, tensor, src):
        parties = self._get_parties_by_rank(src)
        logger.debug(f"[{self.ctx.local}]receiving, index={index}, src={src}, parties={parties}")
        got_tensor = parties.get(self.namespace.indexed_ns(index).federation_tag)[0]
        tensor.copy_(got_tensor)
        return tensor

    def send_many(self, index, tensor, dst_list):
        parties = self._get_parties_by_ranks(dst_list)
        logger.debug(f"[{self.ctx.local}]sending, index={index}, dst={dst_list}, parties={parties}")
        parties.put(self.namespace.indexed_ns(index).federation_tag, tensor)


class WrapCommunicator:
    def __init__(
        self,
        ctx: Context,
        namespace: NS,
        rank_to_party,
        rank,
        world_size,
    ):
        self.namespace = namespace
        self.rank_to_party = rank_to_party

        self.rank = rank
        self.world_size = world_size
        self.federation = MPCFederation(
            rank_to_party=rank_to_party,
            rank=rank,
            world_size=world_size,
            namespace=namespace,
            ctx=ctx,
        )

        self._tensor_send_index = -1
        self._tensor_recv_index = -1

        self._pool = ThreadPoolExecutor(max_workers=2)

    def isend(self, tensor: torch.Tensor, dst):
        self._tensor_send_index += 1

        feature = self._pool.submit(self.federation.send, self._tensor_send_index, tensor, dst)
        return WaitableFuture(feature, f"send_{self._tensor_send_index}_{dst}")

    def irecv(self, tensor, src):
        self._tensor_recv_index += 1

        future = self._pool.submit(self.federation.recv, self._tensor_recv_index, tensor, src)
        return WaitableFuture(future, f"recv_{self._tensor_recv_index}_{src}")

    def send(self, tensor, dst):
        self._tensor_send_index += 1
        self.federation.send(self._tensor_send_index, tensor, dst)

    def recv(self, tensor, src):
        self._tensor_recv_index += 1
        self.federation.recv(self._tensor_recv_index, tensor, src)

    def broadcast(self, tensor, src, group=None, async_op=False):
        if src == self.rank:
            self._tensor_send_index += 1
            self.federation.send_many(
                index=self._tensor_send_index,
                tensor=tensor,
                dst_list=[rank for rank in range(self.world_size) if rank != self.rank],
            )
        else:
            self._tensor_recv_index += 1
            self.federation.recv(
                index=self._tensor_recv_index,
                tensor=tensor,
                src=src,
            )
        return tensor

    def all_gather(self, tensor, async_op=False):
        if async_op:
            raise NotImplementedError()

        self._tensor_send_index += 1
        self.federation.send_many(
            index=self._tensor_send_index,
            tensor=tensor,
            dst_list=[rank for rank in range(self.world_size) if rank != self.rank],
        )
        # self.barrier.wait()
        self._tensor_recv_index += 1
        result = []
        for i in range(self.world_size):
            if i == self.rank:
                result.append(tensor.clone())
            else:
                result.append(
                    self.federation.recv(
                        index=self._tensor_recv_index,
                        tensor=tensor.clone(),
                        src=i,
                    )
                )
        return result

    def all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        if async_op:
            raise NotImplementedError()

        ag = self.all_gather(tensor)
        if op == ReduceOp.SUM:
            return torch.sum(torch.stack(ag), dim=0)
        elif op == torch.distributed.ReduceOp.BXOR:
            return functools.reduce(torch.bitwise_xor, ag)
        else:
            raise NotImplementedError(f"op {op} is not implemented")

    def batched_all_reduce(self, tensor, op=ReduceOp.SUM, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        if async_op:
            raise NotImplementedError()

        ag = self.all_gather(tensor)
        if op == ReduceOp.SUM:
            return torch.sum(torch.stack(ag), dim=0)
        elif op == torch.distributed.ReduceOp.BXOR:
            return functools.reduce(torch.bitwise_xor, ag)
        else:
            raise NotImplementedError()


class WaitableFuture:
    def __init__(self, future, tag):
        self.future = future
        self.tag = tag

    def wait(self):
        self.future.result()
        logger.info(f"wait {self.tag} done")
