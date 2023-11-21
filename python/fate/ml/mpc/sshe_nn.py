import logging

import torch

from fate.arch import Context
from fate.arch.protocol.mpc.nn.sshe.nn_layer import SSHENeuralNetworkAggregatorLayer, SSHENeuralNetworkOptimizerSGD
from . import MPCModule

logger = logging.getLogger(__name__)


class SSHENN(MPCModule):
    def __init__(
        self,
    ):
        ...

    def fit(self, ctx: Context) -> None:
        self.validate_run(ctx)

    def validate_run(self, ctx: Context):
        lr = 0.05
        num_samples = 3
        in_features_a = 4
        in_features_b = 5
        out_features = 2
        ha = torch.rand(num_samples, in_features_a, requires_grad=True, generator=torch.Generator().manual_seed(0))
        hb = torch.rand(num_samples, in_features_b, requires_grad=True, generator=torch.Generator().manual_seed(1))
        h = ctx.mpc.cond_call(lambda: ha, lambda: hb, dst=0)

        generator = torch.Generator().manual_seed(0)
        layer = SSHENeuralNetworkAggregatorLayer(
            ctx,
            in_features_a=in_features_a,
            in_features_b=in_features_b,
            out_features=out_features,
            rank_a=0,
            rank_b=1,
            generator=generator,
        )
        optimizer = SSHENeuralNetworkOptimizerSGD(ctx, layer.parameters(), lr=lr)
        z = layer(h)
        ctx.mpc.info(f"forward={z}", dst=[1])
        ctx.mpc.info(z)
        z2 = z.exp()
        loss = z2.sum()
        loss.backward()

        optimizer.step()
        ctx.mpc.info(f"after backward:\nwa={layer.get_wa()}\nwb={layer.get_wb()}", dst=[1])
        ctx.mpc.info(f"ha.grad={h.grad}", dst=[0])
        ctx.mpc.info(f"hb.grad={h.grad}", dst=[1])

        import time

        time.sleep(3)
        ctx.mpc.info(f"==================ground truth==================")
        ha = torch.rand(num_samples, in_features_a, requires_grad=True, generator=torch.Generator().manual_seed(0))
        hb = torch.rand(num_samples, in_features_b, requires_grad=True, generator=torch.Generator().manual_seed(1))
        wa = torch.rand(in_features_a, out_features, requires_grad=True, generator=torch.Generator().manual_seed(0))
        wb = torch.rand(in_features_b, out_features, requires_grad=True, generator=torch.Generator().manual_seed(0))
        z = ha @ wa + hb @ wb
        ctx.mpc.info(f"forward: {z}")
        z2 = z.exp()
        loss = z2.sum()
        loss.backward(retain_graph=False)
        wa = wa - lr * wa.grad
        wb = wb - lr * wb.grad
        ctx.mpc.info(f"after:\nwa={wa}\nwb={wb}")
        ctx.mpc.info(f"ha.grad={ha.grad}", dst=[0])
        ctx.mpc.info(f"hb.grad={hb.grad}", dst=[1])
