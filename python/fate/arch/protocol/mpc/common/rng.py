#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from fate.arch.protocol import mpc
from fate.arch.protocol.mpc.cuda import CUDALongTensor
from fate.arch.tensor.distributed._tensor import _ShardingShapes, DTensor


def _next_seed_from_generator(_g):
    return torch.randint(1 - 2**63, 2**63 - 1, size=[], generator=_g).item()


def generate_random_ring_element(ctx, size, ring_size=(2**64), generator=None, **kwargs):
    """Helper function to generate a random number from a signed ring"""
    device = kwargs.get("device", torch.device("cpu"))
    device = torch.device("cpu") if device is None else device
    device = torch.device(device) if isinstance(device, str) else device
    if generator is None:
        generator = mpc.generators["local"][device]
    # TODO (brianknott): Check whether this RNG contains the full range we want.
    if isinstance(size, _ShardingShapes):
        shape_and_states = [(shape, _next_seed_from_generator(generator)) for shape in size.shapes]
        rand_element = DTensor.from_sharding_table(
            data=ctx.computing.parallelize(shape_and_states, include_key=False).mapValues(
                lambda x: generate_random_ring_element_by_seed(x[0], seed=x[1], device=device)
            ),
            shapes=size.shapes,
            axis=size.axis,
            dtype=torch.long,
            device=device,
        )
        return rand_element

    else:
        rand_element = torch.randint(
            -(ring_size // 2),
            (ring_size - 1) // 2,
            size,
            generator=generator,
            dtype=torch.long,
            **kwargs,
        )
        if rand_element.is_cuda:
            return CUDALongTensor(rand_element)
        return rand_element


def generate_random_ring_element_by_seed(size, seed, ring_size=(2**64), **kwargs):
    """Helper function to generate a random number from a signed ring"""
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    rand_element = torch.randint(
        -(ring_size // 2),
        (ring_size - 1) // 2,
        size,
        generator=generator,
        dtype=torch.long,
        **kwargs,
    )
    if rand_element.is_cuda:
        return CUDALongTensor(rand_element)
    return rand_element


def generate_kbit_random_tensor(size, bitlength=None, generator=None, **kwargs):
    """Helper function to generate a random k-bit number"""
    if bitlength is None:
        bitlength = torch.iinfo(torch.long).bits
    if bitlength == 64:
        return generate_random_ring_element(size, generator=generator, **kwargs)
    if generator is None:
        device = kwargs.get("device", torch.device("cpu"))
        device = torch.device("cpu") if device is None else device
        device = torch.device(device) if isinstance(device, str) else device
        generator = mpc.generators["local"][device]
    rand_tensor = torch.randint(0, 2**bitlength, size, generator=generator, dtype=torch.long, **kwargs)
    if rand_tensor.is_cuda:
        return CUDALongTensor(rand_tensor)
    return rand_tensor
