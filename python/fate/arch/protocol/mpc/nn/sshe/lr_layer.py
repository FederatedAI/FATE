from typing import Any

import torch

from fate.arch.context import Context
from fate.arch.utils.trace import auto_trace
from fate.arch.protocol.mpc.common.encoding import IgnoreEncodings


class SSHEAggregatorLayer(torch.nn.Module):
    def __init__(
        self, ctx: Context, in_features_a, in_features_b, out_features, rank_a, rank_b, lr=0.05, generator=None
    ):
        self.aggregator = SSHEAggregator(
            ctx, in_features_a, in_features_b, out_features, rank_a, rank_b, lr, generator=generator
        )
        super().__init__()

    def set_lr(self, lr):
        self.aggregator.learning_rate = lr

    def forward(self, x, y):
        virtual_input = torch.zeros(x.shape, requires_grad=True)
        return SSHEAggregatorFunction.apply(virtual_input, x, y, self.aggregator)

    def get_wa(self):
        return self.aggregator.wa.get_plain_text()

    def get_wb(self):
        return self.aggregator.wb.get_plain_text()


class SSHEAggregatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, virtual_input, x, y, aggregator: "SSHEAggregator"):
        ctx.input = x
        output = aggregator.forward(x, y)
        ctx.output = output
        ctx.aggregator = aggregator

        # return virtual output so that backward can be called
        return torch.zeros(1, device="meta")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        aggregator: "SSHEAggregator" = ctx.aggregator
        aggregator.ctx.mpc.info(f"grad_outputs={grad_outputs}", dst=[0, 1])
        ha, hb = aggregator.ctx.mpc.split_variable(ctx.input, aggregator.rank_a, aggregator.rank_b)
        dz = ctx.output
        aggregator.backward(dz, ha, hb)

        return None, None, None, None


class SSHEAggregator:
    def __init__(
        self,
        ctx: Context,
        in_features_a,
        in_features_b,
        out_features,
        rank_a,
        rank_b,
        lr,
        precision_bits=None,
        generator=None,
    ):
        self.ctx = ctx
        self.wa = ctx.mpc.random_tensor(shape=(in_features_a, out_features), src=rank_a, generator=generator)
        self.wb = ctx.mpc.random_tensor(shape=(in_features_b, out_features), src=rank_b, generator=generator)
        self.phe_cipher = ctx.cipher.phe.setup()
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.lr = lr
        self.precision_bits = precision_bits

    @auto_trace(annotation="[z|rank_b] = [xa|rank_a] * <wa> + [xb|rank_b] * <wb>")
    def forward(self, x, y):
        xa = x if self.ctx.rank == self.rank_a else None
        xb = x if self.ctx.rank == self.rank_b else None
        s = self.ctx.mpc.sshe.cross_smm(
            ctx=self.ctx,
            xa=xa,
            xb=xb,
            wa=self.wa,
            wb=self.wb,
            rank_a=self.rank_a,
            rank_b=self.rank_b,
            phe_cipher=self.phe_cipher,
            precision_bits=self.precision_bits,
        )
        z = 0.25 * s - 0.5
        if self.ctx.rank == self.rank_b:
            z = z - y

        return z

    @auto_trace
    def backward(self, d, xa, xb):
        from fate.arch.protocol.mpc.mpc import FixedPointEncoder

        encoder = FixedPointEncoder(self.precision_bits)
        xa_encoded_t = self.ctx.mpc.cond_call(lambda: encoder.encode(xa).T, lambda: None, dst=self.rank_a)
        xb_encoded_t = self.ctx.mpc.cond_call(lambda: encoder.encode(xb).T, lambda: None, dst=self.rank_b)

        # update wa
        # <d.T> @ [xa|rank_a]
        ga = self.ctx.mpc.sshe.smm_mpc_tensor(
            self.ctx,
            op=lambda a, b: b.matmul(a),
            mpc_tensor=d,
            rank_1=self.rank_a,
            tensor_1=xa_encoded_t,
            rank_2=self.rank_b,
            cipher_2=self.phe_cipher,
        )
        with IgnoreEncodings([ga]):
            ga = ga.div_(encoder.scale)

        # <d.T> @ [xb|rank_b]
        gb = self.ctx.mpc.sshe.smm_mpc_tensor(
            self.ctx,
            op=lambda a, b: b.matmul(a),
            mpc_tensor=d,
            rank_1=self.rank_b,
            tensor_1=xb_encoded_t,
            rank_2=self.rank_a,
            cipher_2=self.phe_cipher,
        )
        with IgnoreEncodings([gb]):
            gb = gb.div_(encoder.scale)

        lr = self.lr / d.share.shape[0]
        self.wa -= lr * ga
        self.wb -= lr * gb
        self.ctx.mpc.info(f"ga={self.wa.get_plain_text()}, gb={self.wb.get_plain_text()}", dst=[0])
