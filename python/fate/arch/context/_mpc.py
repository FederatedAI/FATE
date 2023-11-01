import os
import typing
import warnings

import torch

from fate.arch.protocol.mpc.communicator import Communicator
from fate.arch.tensor import mpc
from fate.arch.tensor.mpc.cryptensor import CrypTensor
import logging

if typing.TYPE_CHECKING:
    from fate.arch.context import Context
logger = logging.getLogger(__name__)


class MPC:
    def __init__(self, ctx: "Context"):
        self._ctx = ctx

    @property
    def rank(self):
        return Communicator.get().get_rank()

    @property
    def world_size(self):
        return Communicator.get().get_world_size()

    def init(self):
        """
        Initializes the MPCTensor module by initializing the default provider
        and setting up the RNG generators.
        """
        if Communicator.is_initialized():
            warnings.warn("CrypTen is already initialized.", RuntimeWarning)
            return

        # Initialize communicator
        Communicator.initialize(self._ctx, init_ttp=ttp_required())

        # # Setup party name for file save / load
        # if party_name is not None:
        #     comm.get().set_name(party_name)
        #
        # Setup seeds for Random Number Generation
        if Communicator.get().get_rank() < Communicator.get().get_world_size():
            from fate.arch.protocol.mpc import generators

            _setup_prng(self._ctx, generators)
            if ttp_required():
                from fate.arch.protocol.mpc.provider.ttp_provider import TTPClient

                TTPClient._init()

    @property
    def communicator(self):
        return Communicator.get()

    def cryptensor(self, *args, cryptensor_type=None, **kwargs):
        return mpc.cryptensor(self._ctx, *args, cryptensor_type=cryptensor_type, **kwargs)

    @classmethod
    def is_encrypted_tensor(cls, obj):
        """
        Returns True if obj is an encrypted tensor.
        """
        return isinstance(obj, CrypTensor)

    def print(self, message, dst=[0], print_func=None):
        if print_func is None:
            print_func = print
        if self.rank in dst:
            print_func(message)

    def info(self, message, dst=[0]):
        if isinstance(dst, int):
            dst = [dst]
        if self.rank in dst:
            logger.info(msg=message, stacklevel=2)

    def debug(self, message, dst=[0]):
        if isinstance(dst, int):
            dst = [dst]
        if self.rank in dst:
            logger.debug(msg=message, stacklevel=2)

    def warning(self, message, dst=[0]):
        if isinstance(dst, int):
            dst = [dst]
        if self.rank in dst:
            logger.warning(msg=message, stacklevel=2)

    def error(self, message, dst=[0]):
        if isinstance(dst, int):
            dst = [dst]
        if self.rank in dst:
            logger.error(msg=message, stacklevel=2)

    def cond_call(self, func1, func2=None, dst=0):
        """
        Calls func1 if rank == dst, otherwise calls func2.
        """
        if self.rank == dst:
            return func1()
        else:
            return func2() if func2 is not None else None


def ttp_required():
    from fate.arch.tensor.mpc.config import cfg

    return cfg.mpc.provider == "TTP"


def _setup_prng(ctx: "Context", generators):
    """
    Generate shared random seeds to generate pseudo-random sharings of
    zero. For each device, we generate four random seeds:
        "prev"  - shared seed with the previous party
        "next"  - shared seed with the next party
        "local" - seed known only to the local party (separate from torch's default seed to prevent interference from torch.manual_seed)
        "global"- seed shared by all parties

    The "prev" and "next" random seeds are shared such that each process shares
    one seed with the previous rank process and one with the next rank.
    This allows for the generation of `n` random values, each known to
    exactly two of the `n` parties.

    For arithmetic sharing, one of these parties will add the number
    while the other subtracts it, allowing for the generation of a
    pseudo-random sharing of zero. (This can be done for binary
    sharing using bitwise-xor rather than addition / subtraction)
    """

    # Initialize RNG Generators
    for key in generators.keys():
        generators[key][torch.device("cpu")] = torch.Generator(device=torch.device("cpu"))

    if torch.cuda.is_available():
        cuda_device_names = ["cuda"]
        for i in range(torch.cuda.device_count()):
            cuda_device_names.append(f"cuda:{i}")
        cuda_devices = [torch.device(name) for name in cuda_device_names]

        for device in cuda_devices:
            for key in generators.keys():
                generators[key][device] = torch.Generator(device=device)

    # Generate random seeds for Generators
    # NOTE: Chosen seed can be any number, but we choose as a random 64-bit
    # integer here so other parties cannot guess its value. We use os.urandom(8)
    # here to generate seeds so that forked processes do not generate the same seed.

    # Generate next / prev seeds.
    seed = int.from_bytes(os.urandom(8), "big") - 2**63
    next_seed = torch.tensor(seed)

    # Create local seed - Each party has a separate local generator
    local_seed = int.from_bytes(os.urandom(8), "big") - 2**63

    # Create global generator - All parties share one global generator for sync'd rng
    global_seed = int.from_bytes(os.urandom(8), "big") - 2**63
    global_seed = torch.tensor(global_seed)

    _sync_seeds(ctx, generators, next_seed, local_seed, global_seed)


def _sync_seeds(ctx: "Context", generators, next_seed, local_seed, global_seed):
    """
    Sends random seed to next party, recieve seed from prev. party, and broadcast global seed

    After seeds are distributed. One seed is created for each party to coordinate seeds
    across cuda devices.
    """

    # Populated by recieving the previous party's next_seed (irecv)
    prev_seed = torch.tensor([0], dtype=torch.long)

    # Send random seed to next party, receive random seed from prev party
    world_size = ctx.mpc.communicator.get_world_size()
    rank = ctx.mpc.communicator.get_rank()
    if world_size >= 2:  # Guard against segfaults when world_size == 1.
        next_rank = (rank + 1) % world_size
        prev_rank = (next_rank - 2) % world_size

        req0 = ctx.mpc.communicator.isend(next_seed, next_rank)
        req1 = ctx.mpc.communicator.irecv(prev_seed, src=prev_rank)

        req0.wait()
        req1.wait()
    else:
        prev_seed = next_seed

    prev_seed = prev_seed.item()
    next_seed = next_seed.item()

    # Broadcast global generator - All parties share one global generator for sync'd rng
    global_seed = ctx.mpc.communicator.broadcast(global_seed, 0).item()

    # Create one of each seed per party
    # Note: This is configured to coordinate seeds across cuda devices
    # so that we can one party per gpu. If we want to support configurations
    # where each party runs on multiple gpu's across machines, we will
    # need to modify this.
    for device in generators["prev"].keys():
        generators["prev"][device].manual_seed(prev_seed)
        generators["next"][device].manual_seed(next_seed)
        generators["local"][device].manual_seed(local_seed)
        generators["global"][device].manual_seed(global_seed)
