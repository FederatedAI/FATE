#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import warnings
from fate.arch.tensor.mpc.config import cfg
from . import primitives, provider
from .communicator.communicator import Communicator
from .mpc import MPCTensor
from .ptype import ptype

# Setup RNG generators
generators = {
    "prev": {},
    "next": {},
    "local": {},
    "global": {},
}


__all__ = [
    "MPCTensor",
    "primitives",
    "provider",
    "ptype",
]

# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary

# Set provider
__SUPPORTED_PROVIDERS = {
    "TFP": provider.TrustedFirstParty(),
    "TTP": provider.TrustedThirdParty(),
    "HE": provider.HomomorphicProvider(),
}


def get_default_provider():
    return __SUPPORTED_PROVIDERS[cfg.mpc.provider]


def ttp_required():
    return cfg.mpc.provider == "TTP"


def _setup_prng():
    """
    Generate shared random seeds to generate pseudo-random sharings of
    zero. For each device, we generator four random seeds:
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
    global generators

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

    _sync_seeds(next_seed, local_seed, global_seed)


def _sync_seeds(next_seed, local_seed, global_seed):
    """
    Sends random seed to next party, recieve seed from prev. party, and broadcast global seed

    After seeds are distributed. One seed is created for each party to coordinate seeds
    across cuda devices.
    """
    global generators

    # Populated by recieving the previous party's next_seed (irecv)
    prev_seed = torch.tensor([0], dtype=torch.long)

    # Send random seed to next party, receive random seed from prev party
    world_size = Communicator.get().get_world_size()
    rank = Communicator.get().get_rank()
    if world_size >= 2:  # Guard against segfaults when world_size == 1.
        next_rank = (rank + 1) % world_size
        prev_rank = (next_rank - 2) % world_size

        req0 = Communicator.get().isend(next_seed, next_rank)
        req1 = Communicator.get().irecv(prev_seed, src=prev_rank)

        req0.wait()
        req1.wait()
    else:
        prev_seed = next_seed

    prev_seed = prev_seed.item()
    next_seed = next_seed.item()

    # Broadcase global generator - All parties share one global generator for sync'd rng
    global_seed = Communicator.get().broadcast(global_seed, 0).item()

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


def init(ctx):
    """
    Initializes the MPCTensor module by initializing the default provider
    and setting up the RNG generators.
    """
    if Communicator.is_initialized():
        warnings.warn("CrypTen is already initialized.", RuntimeWarning)
        return

    # Initialize communicator
    Communicator.initialize(ctx, init_ttp=ttp_required())

    # # Setup party name for file save / load
    # if party_name is not None:
    #     comm.get().set_name(party_name)
    #
    # Setup seeds for Random Number Generation
    if Communicator.get().get_rank() < Communicator.get().get_world_size():
        _setup_prng()
        if ttp_required():
            from fate.arch.protocol.mpc.provider.ttp_provider import TTPClient

            TTPClient._init()
