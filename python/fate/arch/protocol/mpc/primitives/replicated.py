#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This file implements Replicated Secret Sharing protocols
# from the CryptGPU repo

import torch

from .. import communicator as comm


def replicate_shares(share_list):
    world_size = comm.get().get_world_size()
    if world_size < 3:
        raise ValueError("Cannot utilize Replicated Sharing securely with < 3 parties.")
    rank = comm.get().get_rank()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    reqs = []
    rep_shares = []
    for share in share_list:
        rep_shares.append(torch.zeros_like(share))

        send_req = comm.get().isend(share.contiguous(), dst=next_rank)
        recv_req = comm.get().irecv(rep_shares[-1], src=prev_rank)

        reqs.extend([send_req, recv_req])

    for req in reqs:
        req.wait()

    # Order [(x1, x2), (y1, y2), ...]
    shares = [(share_list[i], rep_shares[i]) for i in range(len(share_list))]

    return shares


def __replicated_secret_sharing_protocol(op, x, y, *args, **kwargs):
    """Implements bilinear functions using replicated secret shares.
    Shares are input as ArithmeticSharedTensors and are replicated
    within this function to perform computations.

    The protocol used here is that of section 3.2 of ABY3
    (https://eprint.iacr.org/2018/403.pdf).
    """
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    x_shares, y_shares = replicate_shares([x.share, y.share])
    x1, x2 = x_shares
    y1, y2 = y_shares

    z = x.shallow_copy()
    z.share = getattr(torch, op)(x1, y1, *args, **kwargs)
    z.share += getattr(torch, op)(x1, y2, *args, **kwargs)
    z.share += getattr(torch, op)(x2, y1, *args, **kwargs)

    return z


def mul(x, y):
    return __replicated_secret_sharing_protocol("mul", x, y)


def matmul(x, y):
    return __replicated_secret_sharing_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    (x_shares,) = replicate_shares([x.share])
    x1, x2 = x_shares

    x_square = x1**2 + 2 * x1 * x2

    z = x.shallow_copy()
    z.share = x_square
    return z


def truncate(x, y):
    """Protocol to divide an ArithmeticSharedTensor `x` by a constant integer `y`
    using RSS (see ABY3 Figure 2: https://eprint.iacr.org/2018/403.pdf).

    Note: This is currently supported under 3PC only. This is because the protocol
    requires 2-out-of-N secret sharing since only 2 parties can perform division to
    provide statistical guarantees equivalent to 2-out-of-2 truncation.
    """
    if comm.get().get_world_size() != 3:
        raise NotImplementedError("RSS truncation is only implemented for world_size == 3.")

    rank = x.rank

    if rank == 0:
        x.share = x.share.div(y, rounding_mode="trunc")
    elif rank == 1:
        x2 = comm.get().recv(x.share, 2)
        x.share = x.share.add(x2).div(y, rounding_mode="trunc")
    elif rank == 2:
        comm.get().send(x.share, 1)
        x.share -= x.share

    # Add PRZS - this takes the place of r
    x.share += x.PRZS(x.size(), device=x.device).share

    return x


def AND(x, y):
    from .binary import BinarySharedTensor

    x_share = x
    y_share = y
    if isinstance(x, BinarySharedTensor):
        x_share = x.share
        y_share = y.share

    x_shares, y_shares = replicate_shares([x_share, y_share])
    x1, x2 = x_shares
    y1, y2 = y_shares

    z = x.shallow_copy()
    z.share = (x1 & y1) ^ (x2 & y1) ^ (x1 & y2)

    return z
