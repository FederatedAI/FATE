#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math

import torch

# Cache masks and constants to skip computation during each call
__BITS = torch.iinfo(torch.long).bits
__LOG_BITS = int(math.log2(torch.iinfo(torch.long).bits))


@functools.lru_cache(maxsize=None)
def __SPK_circuit_constants(device):
    """
    Generate the __MASKS, __OUT_MASKS, and __MULTIPLIERS constants
    used by __SPK_circuit.
    """
    # Cached SPK masks are:
    # [0] -> 010101010101....0101  =                       01 x 32
    # [1] -> 001000100010....0010  =                     0010 x 16
    # [2] -> 000010000000....0010  =                 00001000 x  8
    # [n] -> [2^n 0s, 1, (2^n -1) 0s] x (32 / (2^n))
    __MASKS = torch.tensor(
        [
            6148914691236517205,
            2459565876494606882,
            578721382704613384,
            36029346783166592,
            140737488388096,
            2147483648,
        ],
        dtype=torch.long,
        device=device,
    )

    __MULTIPLIERS = torch.tensor(
        [(1 << (2**iter + 1)) - 2 for iter in range(__LOG_BITS)], device=device
    )
    __OUT_MASKS = __MASKS * __MULTIPLIERS

    return __MASKS, __OUT_MASKS, __MULTIPLIERS


def __SPK_circuit(S, P):
    """
    Computes the Set-Propagate-Kill Tree circuit for a set (S, P)
    (K is implied by S, P since (SPK) is one-hot)

    (See section 6.3 of Damgard, "Unconditionally Secure Constant-Rounds
    Multi-Party Computation for Equality, Comparison, Bits and Exponentiation")

    At each stage:
        S <- S0 ^ (P0 & S1)
        P <- P0 & P1
        K <- K0 ^ (P0 & K1) <- don't need K since it is implied by S and P
    """
    from .binary import BinarySharedTensor

    # Vectorize private AND calls to reduce rounds:
    SP = BinarySharedTensor.stack([S, P])

    __MASKS, __OUT_MASKS, __MULTIPLIERS = __SPK_circuit_constants(SP.device)

    # fmt: off
    # Tree reduction circuit
    for i in range(__LOG_BITS):
        in_mask = __MASKS[i]                # Start of arrows
        out_mask = __OUT_MASKS[i]           # End of arrows
        not_out_mask = out_mask ^ -1        # Not (end of arrows)

        # Set up S0, S1, P0, and P1
        P0 = SP[1] & out_mask               # Mask P0 from P
        S1P1 = SP & in_mask                 # Mask S1P1 from SP
        S1P1._tensor *= __MULTIPLIERS[i]    # Fan out S1P1 along arrows

        # Update S and P
        update = P0 & S1P1                  # S0 ^= P0 & S1, P0 = P0 & P1
        SP[1] &= not_out_mask
        SP ^= update
    # fmt: on
    return SP[0], SP[1]


def __P_circuit(P):
    """
    Computes the Propagate Tree circuit for input P.
    The P circuit will return 1 only if the binary of
    the input is all ones (i.e. the value is -1).

    Otherwise this circuit returns 0

    At each stage:
        P <- P0 & P1
    """
    shift = __BITS // 2
    for _ in range(__LOG_BITS):
        P &= P << shift  # using lshift since rshift was modified to arithmetic
        shift //= 2
    return P


def __flip_sign_bit(x):
    return x ^ -(2**63)


def __get_sign_bit(x):
    from .binary import BinarySharedTensor

    y = x >> 63

    # NOTE: __rshift__ was changed to arithmetic shift
    if isinstance(y, BinarySharedTensor):
        y.share = y.share.eq(-1).long()
    else:
        y = y.eq(-1).long()
    return y


def add(x, y):
    """Returns x + y from BinarySharedTensors `x` and `y`"""
    S = x & y
    P = x ^ y
    carry, _ = __SPK_circuit(S, P)
    return P ^ (carry << 1)


def eq(x, y):
    """Returns x == y from BinarySharedTensors `x` and `y`"""
    bitwise_equal = ~(x ^ y)
    P = __P_circuit(bitwise_equal)
    return __get_sign_bit(P)


def lt(x, y):
    """Returns x < y from BinarySharedTensors `x` and `y`"""
    x, y = __flip_sign_bit(x), __flip_sign_bit(y)

    S = y & ~x
    P = ~(x ^ y)
    S, _ = __SPK_circuit(S, P)
    return __get_sign_bit(S)


def le(x, y):
    """Returns x <= y from BinarySharedTensors `x` and `y`"""
    x, y = __flip_sign_bit(x), __flip_sign_bit(y)

    S = y & ~x
    P = ~(x ^ y)
    S, P = __SPK_circuit(S, P)
    return __get_sign_bit(S ^ P)


def gt(x, y):
    """Returns x > y from BinarySharedTensors `x` and `y`"""
    x, y = __flip_sign_bit(x), __flip_sign_bit(y)

    S = x & ~y
    P = ~(x ^ y)
    S, _ = __SPK_circuit(S, P)
    return __get_sign_bit(S)


def ge(x, y):
    """Returns x >= y from BinarySharedTensors `x` and `y`"""
    x, y = __flip_sign_bit(x), __flip_sign_bit(y)

    S = x & ~y
    P = ~(x ^ y)
    S, P = __SPK_circuit(S, P)
    return __get_sign_bit(S ^ P)
