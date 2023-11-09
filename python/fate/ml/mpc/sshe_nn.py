import logging


import torch

from . import MPCModule
from ...arch import Context
from ...arch.tensor import DTensor
from .mpc_sa_layer import SSHEAggregatorLayer

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
        ha = torch.rand(10, 4, requires_grad=True, generator=torch.Generator().manual_seed(0))
        hb = torch.rand(10, 5, requires_grad=True, generator=torch.Generator().manual_seed(1))
        h = ctx.mpc.cond_call(lambda: ha, lambda: hb, dst=0)

        generator = torch.Generator().manual_seed(0)
        layer = SSHEAggregatorLayer(
            ctx, in_features_a=4, in_features_b=5, out_features=2, rank_a=0, rank_b=1, lr=lr, generator=generator
        )
        ctx.mpc.info(f"before:\nwa={layer.get_wa()}\nwb={layer.get_wb()}")
        z = layer(h)
        ctx.mpc.info(z)
        loss = (z * 0.001).sum()
        loss.backward()
        ctx.mpc.info(f"after:\nwa={layer.get_wa()}\nwb={layer.get_wb()}")
        ctx.mpc.info(f"h.grad={h.grad}", dst=[0, 1])

        ctx.mpc.info(f"==================ground truth==================")
        ha = torch.rand(10, 4, requires_grad=True, generator=torch.Generator().manual_seed(0))
        hb = torch.rand(10, 5, requires_grad=True, generator=torch.Generator().manual_seed(1))
        wa = torch.rand(4, 2, requires_grad=True, generator=torch.Generator().manual_seed(0))
        wb = torch.rand(5, 2, requires_grad=True, generator=torch.Generator().manual_seed(0))
        ctx.mpc.info(f"before:\nwa={wa}\nwb={wb}")
        z = torch.matmul(ha, wa) + torch.matmul(hb, wb)
        loss = (z * 0.001).sum()
        loss.backward()
        wa = wa - lr * wa.grad
        wb = wb - lr * wb.grad
        ctx.mpc.info(f"after:\nwa={wa}\nwb={wb}")
        ctx.mpc.info(f"ha.grad={ha.grad}\nhb.grad={hb.grad}")
