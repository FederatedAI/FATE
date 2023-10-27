#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from fate.arch.tensor import mpc
from . import MPCModule
from ...arch import Context
from ...arch.tensor import DTensor


class Toy(MPCModule):
    def __init__(
        self,
    ):
        ...

    def fit(self, ctx: Context) -> None:
        if ctx.mpc.rank == 0:
            x_alice = DTensor.from_sharding_list(ctx, [torch.rand(5, 11), torch.rand(4, 11), torch.rand(3, 11)])
        else:
            x_alice = DTensor.from_sharding_list(ctx, [torch.zeros(5, 11), torch.zeros(4, 11), torch.zeros(3, 11)])

        if ctx.mpc.rank == 1:
            x_bob = DTensor.from_sharding_list(ctx, [torch.rand(5, 11), torch.rand(4, 11), torch.rand(3, 11)])
        else:
            x_bob = DTensor.from_sharding_list(ctx, [torch.zeros(5, 11), torch.zeros(4, 11), torch.zeros(3, 11)])

        x_alice_enc = mpc.cryptensor(x_alice, src=0)
        x_bob_enc = mpc.cryptensor(x_bob, src=1)

        mpc.print((x_alice_enc + x_bob_enc).get_plain_text())
