import logging

import torch

from fate.arch import Context
from ..abc.module import Module
from fate.arch.dataframe import DataFrame
from fate.arch.protocol.mpc.nn.sshe.lr_layer import (
    SSHELogisticRegressionLayer,
    SSHELogisticRegressionLossLayer,
    SSHEOptimizerSGD,
)

logger = logging.getLogger(__name__)


class SSHELogisticRegression(Module):
    def __init__(self, lr=0.05):
        self.lr = lr

    def fit(self, ctx: Context, input_data: DataFrame) -> None:
        rank_a, rank_b = ctx.hosts[0].rank, ctx.guest.rank
        y = ctx.mpc.cond_call(lambda: input_data.label.as_tensor(), lambda: None, dst=rank_b)
        h = input_data.as_tensor()

        layer = SSHELogisticRegressionLayer(
            ctx,
            in_features_a=ctx.mpc.option_call(lambda: h.shape[1], dst=rank_a),
            in_features_b=ctx.mpc.option_call(lambda: h.shape[1], dst=rank_b),
            out_features=1,
            rank_a=rank_a,
            rank_b=rank_b,
            wa_init_fn=lambda shape: torch.rand(shape),
            wb_init_fn=lambda shape: torch.rand(shape),
        )

        loss_fn = SSHELogisticRegressionLossLayer(ctx, rank_a=rank_a, rank_b=rank_b)
        optimizer = SSHEOptimizerSGD(ctx, layer.parameters(), lr=self.lr)

        for i in range(1):
            z = layer(h)
            loss = loss_fn(z, y)
            if i % 3 == 0:
                logger.info(f"loss: {loss.get()}")
            loss.backward()
            optimizer.step()
            wa = layer.wa.get_plain_text()
            wb = layer.wb.get_plain_text()
