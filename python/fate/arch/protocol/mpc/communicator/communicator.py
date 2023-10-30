import functools
import logging
import sys
import timeit
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed import ReduceOp
from torch.distributed import ReduceOp

from fate.arch.context import Context, NS, Parties

logger = logging.getLogger(__name__)


class Communicator:
    """
    FATECommunicator is a wrapper around the FATE communicator.
    """

    instance = None

    def __init__(
        self,
        ctx: Context,
        namespace: NS,
        rank_to_party,
        rank,
        world_size,
    ):
        self.ctx = ctx
        self.namespace = namespace
        self.rank_to_party = rank_to_party
        self.rank = rank
        self.world_size = world_size
        self.main_group = None
        self._tensor_send_index = -1
        self._tensor_recv_index = -1

        self._pool = ThreadPoolExecutor(max_workers=2)

    @classmethod
    def is_initialized(cls):
        return cls.instance is not None

    @classmethod
    def get(cls) -> "Communicator":
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
        world_size = ctx.world_size
        rank = ctx.local.rank
        namespace = NS(ctx.namespace.sub_ns("mpc"), 0)
        rank_to_party = {p.rank: p.party for p in ctx.parties}
        cls.instance = Communicator(
            ctx,
            namespace,
            rank_to_party,
            rank,
            world_size,
        )

    @classmethod
    def shutdown(cls):
        pass

    def send(self, tensor, dst):
        self._tensor_send_index += 1
        self._send(self._tensor_send_index, tensor, dst)

    def recv(self, tensor, src=None):
        self._tensor_recv_index += 1
        self._recv(self._tensor_recv_index, tensor, src)

    def isend(self, tensor, dst):
        self._tensor_send_index += 1

        feature = self._pool.submit(self._send, self._tensor_send_index, tensor, dst)
        return WaitableFuture(feature, f"send_{self._tensor_send_index}_{dst}")

    def irecv(self, tensor: torch.Tensor, src=None):
        self._tensor_recv_index += 1

        future = self._pool.submit(self._recv, self._tensor_recv_index, tensor, src)
        return WaitableFuture(future, f"recv_{self._tensor_recv_index}_{src}")

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
            ag = self.all_gather(input)
            if op == torch.distributed.ReduceOp.SUM:
                return self._sum(ag)
            elif op == torch.distributed.ReduceOp.BXOR:
                return functools.reduce(torch.bitwise_xor, ag)
            else:
                raise NotImplementedError(f"op {op} is not implemented")

    @staticmethod
    def _sum(tensor_list):
        # return torch.sum(torch.stack(ag), dim=0)
        result = tensor_list[0]
        for t in tensor_list[1:]:
            result += t
        return result

    def gather(self, tensor, dst, async_op=False):
        raise NotImplementedError

    def all_gather(self, tensor, async_op=False):
        if async_op:
            raise NotImplementedError()

        self._tensor_send_index += 1
        self._send_many(
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
                    self._recv(
                        index=self._tensor_recv_index,
                        tensor=tensor.clone(),
                        src=i,
                    )
                )
        return result

    def broadcast(self, input, src, group=None, batched=False):
        self._assert_initialized()
        group = self.main_group if group is None else group
        if batched:
            assert isinstance(input, list), "batched reduce input must be a list"
            reqs = []
            for tensor in input:
                reqs.append(self.broadcast(tensor.data, src, group=group, batched=False))
        else:
            assert torch.is_tensor(input.data), "unbatched input for reduce must be a torch tensor"
            if src == self.rank:
                self._tensor_send_index += 1
                self._send_many(
                    index=self._tensor_send_index,
                    tensor=input.data,
                    dst_list=[rank for rank in range(self.world_size) if rank != self.rank],
                )
            else:
                self._tensor_recv_index += 1
                self._recv(
                    index=self._tensor_recv_index,
                    tensor=input.data,
                    src=src,
                )
        return input

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.comm_rounds = 0
        self.comm_bytes = 0
        self.comm_time = 0

    def print_communication_stats(self):
        """
        Prints communication statistics.

        NOTE: Each party performs its own logging of communication, so one needs
        to sum the number of bytes communicated over all parties and divide by
        two (to prevent double-counting) to obtain the number of bytes
        communicated in the overall system.
        """

        logger.info("====Communication Stats====")
        logger.info("Rounds: {}".format(self.comm_rounds))
        logger.info("Bytes: {}".format(self.comm_bytes))
        logger.info("Communication time: {}".format(self.comm_time))

    def get_communication_stats(self):
        """
        Returns communication statistics in a Python dict.

        NOTE: Each party performs its own logging of communication, so one needs
        to sum the number of bytes communicated over all parties and divide by
        two (to prevent double-counting) to obtain the number of bytes
        communicated in the overall system.
        """
        return {
            "rounds": self.comm_rounds,
            "bytes": self.comm_bytes,
            "time": self.comm_time,
        }

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.comm_rounds += 1
        self.comm_bytes += nelement * self.BYTES_PER_ELEMENT

    def _log_communication_time(self, comm_time):
        self.comm_time += comm_time

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

    def _send(self, index, tensor, dst):
        parties = self._get_parties_by_rank(dst)
        logger.debug(f"[{self.ctx.local}]sending, index={index}, dst={dst}, parties={parties}")
        parties.put(self.namespace.indexed_ns(index).federation_tag, tensor)

    def _recv(self, index, tensor, src):
        parties = self._get_parties_by_rank(src)
        logger.debug(f"[{self.ctx.local}]receiving, index={index}, src={src}, parties={parties}")
        got_tensor = parties.get(self.namespace.indexed_ns(index).federation_tag)[0]
        tensor.copy_(got_tensor)
        return tensor

    def _send_many(self, index, tensor, dst_list):
        parties = self._get_parties_by_ranks(dst_list)
        logger.debug(f"[{self.ctx.local}]sending, index={index}, dst={dst_list}, parties={parties}")
        parties.put(self.namespace.indexed_ns(index).federation_tag, tensor)


class WaitableFuture:
    def __init__(self, future, tag):
        self.future = future
        self.tag = tag

    def wait(self):
        self.future.result()
        logger.info(f"wait {self.tag} done")


def _logging(func):
    """
    Decorator that performs logging of communication statistics.

    NOTE: Each party performs its own logging of communication, so one needs to
    sum the number of bytes communicated over all parties and divide by two
    (to prevent double-counting) to obtain the number of bytes communicated in
    the overall system.
    """
    from functools import wraps

    @wraps(func)
    def logging_wrapper(self, *args, **kwargs):
        from fate.arch.tensor.mpc.config import cfg

        # TODO: Replace this
        # - hacks the inputs into some of the functions for world_size 1:
        world_size = self.get_world_size()
        if world_size < 2:
            if func.__name__ in ["gather", "all_gather"]:
                return [args[0]]
            elif len(args) > 0:
                return args[0]

        # only log communication if needed:
        if cfg.communicator.verbose:
            rank = self.get_rank()
            _log = self._log_communication

            # count number of bytes communicates for each MPI collective:
            if func.__name__ == "barrier":
                _log(0)
            elif func.__name__ in ["send", "recv", "isend", "irecv"]:
                _log(args[0].nelement())  # party sends or receives tensor
            elif func.__name__ == "scatter":
                if args[1] == rank:  # party scatters P - 1 tensors
                    nelements = sum(x.nelement() for idx, x in enumerate(args[0]) if idx != rank)
                    _log(nelements)  # NOTE: We deal with other parties later
            elif func.__name__ == "all_gather":
                _log(2 * (world_size - 1) * args[0].nelement())
                # party sends and receives P - 1 tensors
            elif func.__name__ == "send_obj":
                nbytes = sys.getsizeof(args[0])
                _log(nbytes / self.BYTES_PER_ELEMENT)  # party sends object
            elif func.__name__ == "broadcast_obj":
                nbytes = sys.getsizeof(args[0])
                _log(nbytes / self.BYTES_PER_ELEMENT * (world_size - 1))
                # party sends object to P - 1 parties
            elif func.__name__ in ["broadcast", "gather", "reduce"]:
                multiplier = world_size - 1 if args[1] == rank else 1
                # broadcast: party sends tensor to P - 1 parties, or receives 1 tensor
                # gather: party receives P - 1 tensors, or sends 1 tensor
                # reduce: party receives P - 1 tensors, or sends 1 tensor
                if "batched" in kwargs and kwargs["batched"]:
                    nelements = sum(x.nelement() for x in args[0])
                    _log(nelements * multiplier)
                else:
                    _log(args[0].nelement() * multiplier)
            elif func.__name__ == "all_reduce":
                # each party sends and receives one tensor in ring implementation
                if "batched" in kwargs and kwargs["batched"]:
                    nelements = sum(2 * x.nelement() for x in args[0])
                    _log(nelements)
                else:
                    _log(2 * args[0].nelement())

            # execute and time the MPI collective:
            tic = timeit.default_timer()
            result = func(self, *args, **kwargs)
            toc = timeit.default_timer()
            self._log_communication_time(toc - tic)

            # for some function, we only know the object size now:
            if func.__name__ == "scatter" and args[1] != rank:
                _log(result.nelement())  # party receives 1 tensor
            if func.__name__ == "recv_obj":
                _log(sys.getsizeof(result) / self.BYTES_PER_ELEMENT)
                # party receives 1 object

            return result

        return func(self, *args, **kwargs)

    return logging_wrapper
