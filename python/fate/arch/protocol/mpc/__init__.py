#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fate.arch.tensor.mpc.config import cfg
from . import primitives, provider

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
