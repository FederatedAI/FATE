#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import copy
import logging
import os
import typing

import torch

from . import nn
from . import primitives, provider
from .communicator import Communicator
from .config import cfg
from .cryptensor import CrypTensor
from .mpc import MPCTensor
from .ptype import ptype

if typing.TYPE_CHECKING:
    from fate.arch.context import Context

# Setup RNG generators
generators = {
    "prev": {},
    "next": {},
    "local": {},
    "global": {},
}


__all__ = ["MPCTensor", "primitives", "provider", "ptype", "generators"]

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
    return __SUPPORTED_PROVIDERS[cfg.safety.mpc.provider]


def cryptensor(ctx, *args, cryptensor_type=None, **kwargs):
    """
    Factory function to return encrypted tensor of given `cryptensor_type`. If no
    `cryptensor_type` is specified, the default type is used.
    """

    # determine CrypTensor type to use:
    if cryptensor_type is None:
        cryptensor_type = get_default_cryptensor_type()
    if cryptensor_type not in CrypTensor.__CRYPTENSOR_TYPES__:
        raise ValueError("CrypTensor type %s does not exist." % cryptensor_type)

    # create CrypTensor:
    return CrypTensor.__CRYPTENSOR_TYPES__[cryptensor_type](ctx, *args, **kwargs)


def is_encrypted_tensor(obj):
    """
    Returns True if obj is an encrypted tensor.
    """
    return isinstance(obj, CrypTensor)


def get_default_cryptensor_type():
    """Gets the default type used to create `CrypTensor`s."""

    return CrypTensor.__DEFAULT_CRYPTENSOR_TYPE__


def get_cryptensor_type(tensor):
    """Gets the type name of the specified `tensor` `CrypTensor`."""
    if not isinstance(tensor, CrypTensor):
        raise ValueError("Specified tensor is not a CrypTensor: {}".format(type(tensor)))
    for name, cls in CrypTensor.__CRYPTENSOR_TYPES__.items():
        if isinstance(tensor, cls):
            return name
    raise ValueError("Unregistered CrypTensor type: {}".format(type(tensor)))


def load_from_party(
    f=None,
    preloaded=None,
    encrypted=False,
    model_class=None,
    src=0,
    load_closure=torch.load,
    **kwargs,
):
    """
    Loads an object saved with `torch.save()` or `crypten.save_from_party()`.

    Args:
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        preloaded: Use the preloaded value instead of loading a tensor/model from f.
        encrypted: Determines whether crypten should load an encrypted tensor
                      or a plaintext torch tensor.
        model_class: Takes a model architecture class that is being communicated. This
                    class will be considered safe for deserialization so non-source
                    parties will be able to receive a model of this type from the
                    source party.
        src: Determines the source of the tensor. If `src` is None, each
            party will attempt to read in the specified file. If `src` is
            specified, the source party will read the tensor from `f` and it
            will broadcast it to the other parties
        load_closure: Custom load function that matches the interface of `torch.load`,
        to be used when the tensor is saved with a custom save function in
        `crypten.save_from_party`. Additional kwargs are passed on to the closure.
    """

    if encrypted:
        raise NotImplementedError("Loading encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Load failed: src argument must be an integer"
        assert src >= 0 and src < Communicator.get().get_world_size(), "Load failed: src must be in [0, world_size)"

        # source party
        if Communicator.get().get_rank() == src:
            assert (f is None and (preloaded is not None)) or (
                (f is not None) and preloaded is None
            ), "Exactly one of f and preloaded must not be None"

            if f is None:
                result = preloaded
            if preloaded is None:
                result = load_closure(f, **kwargs)

            # Zero out the tensors / modules to hide loaded data from broadcast
            if torch.is_tensor(result):
                result_zeros = result.new_zeros(result.size())
            elif isinstance(result, torch.nn.Module):
                result_zeros = copy.deepcopy(result)
                for p in result_zeros.parameters():
                    p.data.fill_(0)
            else:
                result = Communicator.get().broadcast_obj(-1, src)
                raise TypeError("Unrecognized load type %s" % type(result))

            Communicator.get().broadcast_obj(result_zeros, src)

        # Non-source party
        else:
            if model_class is not None:
                from .common import serial

                serial.register_safe_class(model_class)
            result = Communicator.get().broadcast_obj(None, src)
            if isinstance(result, int) and result == -1:
                raise TypeError("Unrecognized load type from src party")

        if torch.is_tensor(result):
            result = cryptensor(result, src=src)

        # TODO: Encrypt modules before returning them
        # if isinstance(result, torch.nn.Module):
        #     result = crypten.nn.from_pytorch(result, src=src)

        result.src = src
        return result


def load(f, load_closure=torch.load, **kwargs):
    """
    Loads shares from an encrypted object saved with `crypten.save()`
    Args:
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        load_closure: Custom load function that matches the interface of
        `torch.load`, to be used when the tensor is saved with a custom
        save function in `crypten.save`. Additional kwargs are passed on
        to the closure.
    """
    if "src" in kwargs:
        raise SyntaxError("crypten.load() should not be used with `src` argument. Use load_from_party() instead.")

    # TODO: Add support for loading from correct device (kwarg: map_location=device)
    if load_closure == torch.load:
        obj = load_closure(f)
    else:
        obj = load_closure(f, **kwargs)
    return obj


def save_from_party(obj, f, src=0, save_closure=torch.save, **kwargs):
    """
    Saves a CrypTensor or PyTorch tensor to a file.

    Args:
        obj: The CrypTensor or PyTorch tensor to be saved
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        src: The source party that writes data to the specified file.
        save_closure: Custom save function that matches the interface of `torch.save`,
        to be used when the tensor is saved with a custom load function in
        `crypten.load_from_party`. Additional kwargs are passed on to the closure.
    """
    if is_encrypted_tensor(obj):
        raise NotImplementedError("Saving encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Save failed: src must be an integer"
        assert (
            src >= 0 and src < Communicator.get().get_world_size()
        ), "Save failed: src must be an integer in [0, world_size)"

        if Communicator.get().get_rank() == src:
            save_closure(obj, f, **kwargs)

    # Implement barrier to avoid race conditions that require file to exist
    Communicator.get().barrier()


def save(obj, f, save_closure=torch.save, **kwargs):
    """
    Saves the shares of CrypTensor or an encrypted model to a file.

    Args:
        obj: The CrypTensor or PyTorch tensor to be saved
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        save_closure: Custom save function that matches the interface of `torch.save`,
        to be used when the tensor is saved with a custom load function in
        `crypten.load`. Additional kwargs are passed on to the closure.
    """
    # TODO: Add support for saving to correct device (kwarg: map_location=device)
    save_closure(obj, f, **kwargs)
    Communicator.get().barrier()


def where(condition, input, other):
    """
    Return a tensor of elements selected from either `input` or `other`, depending
    on `condition`.
    """
    if is_encrypted_tensor(condition):
        return condition * input + (1 - condition) * other
    elif torch.is_tensor(condition):
        condition = condition.float()
    return input * condition + other * (1 - condition)


def cat(tensors, dim=0):
    """
    Concatenates the specified CrypTen `tensors` along dimension `dim`.
    """
    assert isinstance(tensors, list), "input to cat must be a list"
    if all(torch.is_tensor(t) for t in tensors):
        return torch.cat(tensors)

    assert all(isinstance(t, CrypTensor) for t in tensors), "inputs must be CrypTensors"
    tensor_types = [get_cryptensor_type(t) for t in tensors]
    assert all(
        ttype == tensor_types[0] for ttype in tensor_types
    ), "cannot concatenate CrypTensors with different underlying types"
    if len(tensors) == 1:
        return tensors[0]
    return type(tensors[0]).cat(tensors, dim=dim)


def stack(tensors, dim=0):
    """
    Stacks the specified CrypTen `tensors` along dimension `dim`. In contrast to
    `crypten.cat`, this adds a dimension to the result tensor.
    """
    assert isinstance(tensors, list), "input to stack must be a list"
    assert all(isinstance(t, CrypTensor) for t in tensors), "inputs must be CrypTensors"
    tensor_types = [get_cryptensor_type(t) for t in tensors]
    assert all(
        ttype == tensor_types[0] for ttype in tensor_types
    ), "cannot stack CrypTensors with different underlying types"
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    return type(tensors[0]).stack(tensors, dim=dim)


def rand(*sizes, device=None, cryptensor_type=None):
    """
    Returns a tensor with elements uniformly sampled in [0, 1).
    """
    with CrypTensor.no_grad():
        if cryptensor_type is None:
            cryptensor_type = get_default_cryptensor_type()
        return CrypTensor.__CRYPTENSOR_TYPES__[cryptensor_type].rand(*sizes, device=device)


def randn(*sizes, cryptensor_type=None):
    """
    Returns a tensor with normally distributed elements.
    """
    with CrypTensor.no_grad():
        if cryptensor_type is None:
            cryptensor_type = get_default_cryptensor_type()
        return CrypTensor.__CRYPTENSOR_TYPES__[cryptensor_type].randn(*sizes)


def bernoulli(tensor, cryptensor_type=None):
    """
    Returns a tensor with elements in {0, 1}. The i-th element of the
    output will be 1 with probability according to the i-th value of the
    input tensor.
    """
    return rand(tensor.size(), cryptensor_type=cryptensor_type) < tensor


def __multiprocess_print_helper(print_func, *args, in_order=False, dst=0, **kwargs):
    """
    Helper for print / log functions to reduce copy-pasted code
    """
    # in_order : True
    if in_order:
        for i in range(Communicator.get().get_world_size()):
            if Communicator.get().get_rank() == i:
                print_func(*args, **kwargs)
            Communicator.get().barrier()
        return

    # in_order : False
    if isinstance(dst, int):
        dst = [dst]
    assert isinstance(dst, (list, tuple)), "print destination must be a list or tuple of party ranks"

    if Communicator.get().get_rank() in dst:
        print_func(*args, **kwargs)


def print(*args, in_order=False, dst=0, **kwargs):
    """
    Prints with formatting options that account for multiprocessing. This
    function prints with the output of:

        print(*args, **kwargs)

    Args:
        in_order: A boolean that determines whether to print from one-party only
            or all parties, in order. If True, this function will output from
            party 0 first, then print in order through party N. If False, this
            function will only output from a single party, given by `dst`.
        dst: The destination party rank(s) to output from if `in_order` is False.
            This can be an integer or list of integers denoting a single rank or
            multiple ranks to print from.
    """
    __multiprocess_print_helper(builtins.print, *args, in_order=in_order, dst=dst, **kwargs)


def log(*args, in_order=False, dst=0, **kwargs):
    """
    Logs with formatting options that account for multiprocessing. This
    function logs with the output of:

        logging.log(*args, **kwargs)

    Args:
        in_order: A boolean that determines whether to log from one-party only
            or all parties, in order. If True, this function will output from
            party 0 first, then log in order through party N. If False, this
            function will only output from a single party, given by `dst`.
        dst: The destination party rank(s) to output from if `in_order` is False.
            This can be an integer or list of integers denoting a single rank or
            multiple ranks to log from.
    """
    __multiprocess_print_helper(logging.info, *args, in_order=in_order, dst=dst, **kwargs)


# TupleProvider tracing functions
def trace(tracing=True):
    get_default_provider().trace(tracing=tracing)


def trace_once():
    get_default_provider().trace_once()


def fill_cache():
    get_default_provider().fill_cache()


def ttp_required():
    return cfg.safety.mpc.provider == "TTP"


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
