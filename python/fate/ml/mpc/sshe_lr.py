import logging


import torch

from fate.arch import Context
from ..abc.module import Module
from fate.arch.dataframe import DataFrame
from fate.arch.protocol.mpc.nn.sshe.lr_layer import SSHEAggregator, SSHEAggregatorLayer

logger = logging.getLogger(__name__)


class SSHELR(Module):
    def __init__(self, lr=0.05):
        self.lr = lr

    def fit(self, ctx: Context, input_data: DataFrame) -> None:
        rank_a, rank_b = ctx.hosts[0].rank, ctx.guest.rank
        y = ctx.mpc.cond_call(lambda: input_data.label.as_tensor(), lambda: None, dst=rank_b)
        h = input_data.as_tensor()
        shape_a = ctx.mpc.communicator.broadcast_obj(obj=h.shape if ctx.rank == rank_a else None, src=rank_a)
        shape_b = ctx.mpc.communicator.broadcast_obj(obj=h.shape if ctx.rank == rank_b else None, src=rank_b)
        generator = torch.Generator().manual_seed(0)
        layer = SSHEAggregatorLayer(
            ctx,
            in_features_a=shape_a[1],
            in_features_b=shape_b[1],
            out_features=1,
            rank_a=rank_a,
            rank_b=rank_b,
            lr=self.lr,
            generator=generator,
        )

        for i in range(10):
            z = layer(h, y)
            z.backward()
