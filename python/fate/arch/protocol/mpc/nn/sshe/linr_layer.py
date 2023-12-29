import typing

import torch

from fate.arch.context import Context
from fate.arch.protocol.mpc.common.encoding import IgnoreEncodings
from fate.arch.protocol.mpc.mpc import FixedPointEncoder
from fate.arch.trace import auto_trace


class SSHELinearRegressionLayer:
    def __init__(
        self,
        ctx: Context,
        in_features_a,
        in_features_b,
        out_features,
        rank_a,
        rank_b,
        wa_init_fn: typing.Callable[[typing.Tuple], torch.Tensor],
        wb_init_fn: typing.Callable[[typing.Tuple], torch.Tensor],
        precision_bits=None,
        sync_shape=True,
        cipher_options=None,
    ):
        self.ctx = ctx
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.group = ctx.mpc.communicator.new_group(
            [rank_a, rank_b], f"{ctx.namespace.federation_tag}.sshe_aggregator_layer"
        )

        if sync_shape:
            ctx.mpc.option_assert(in_features_a is not None, "in_features_a must be specified", dst=rank_a)
            ctx.mpc.option_assert(
                in_features_b is None, "in_features_b must be None when sync_shape is True", dst=rank_a
            )
            ctx.mpc.option_assert(in_features_b is not None, "in_features_b must be specified", dst=rank_b)
            ctx.mpc.option_assert(
                in_features_a is None, "in_features_a must be None when sync_shape is True", dst=rank_b
            )
            in_features_a = ctx.mpc.communicator.broadcast_obj(obj=in_features_a, src=rank_a)
            in_features_b = ctx.mpc.communicator.broadcast_obj(obj=in_features_b, src=rank_b)
        else:
            ctx.mpc.option_assert(
                in_features_a is not None, "in_features_a must be specified when sync_shape is False", dst=rank_a
            )
            ctx.mpc.option_assert(
                in_features_b is not None, "in_features_b must be specified when sync_shape is False", dst=rank_b
            )

        self.wa = ctx.mpc.init_tensor(shape=(in_features_a, out_features), init_func=wa_init_fn, src=rank_a)
        self.wb = ctx.mpc.init_tensor(shape=(in_features_b, out_features), init_func=wb_init_fn, src=rank_b)
        self.phe_cipher = ctx.cipher.phe.setup(options=cipher_options)
        self.precision_bits = precision_bits

    @auto_trace(annotation="[z|rank_b] = [xa|rank_a] * <wa> + [xb|rank_b] * <wb>")
    def forward(self, x):
        xa, xb = self.ctx.mpc.split_variable(x, self.rank_a, self.rank_b)
        z = self.ctx.mpc.sshe.cross_smm(
            ctx=self.ctx,
            group=self.group,
            xa=xa,
            xb=xb,
            wa=self.wa,
            wb=self.wb,
            rank_a=self.rank_a,
            rank_b=self.rank_b,
            phe_cipher=self.phe_cipher,
            precision_bits=self.precision_bits,
        )

        # set backward function
        z.backward = SSHELinearRegressionLayerBackwardFunction(
            ctx=self.ctx,
            group=self.group,
            rank_a=self.rank_a,
            rank_b=self.rank_b,
            phe_cipher=self.phe_cipher,
            encoder=FixedPointEncoder(self.precision_bits),
            wa=self.wa,
            wb=self.wb,
            dz=z,
            x=x,
        )
        return z

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.wa, self.wb]


class SSHELinearRegressionLayerBackwardFunction:
    def __init__(self, ctx, group, rank_a, rank_b, phe_cipher, encoder, wa, wb, dz, x):
        self.ctx = ctx
        self.group = group
        self.encoder = encoder
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.phe_cipher = phe_cipher
        self.wa = wa
        self.wb = wb

        self.dz = dz
        self.x = x

    @auto_trace
    def __call__(self, dz):
        xa, xb = self.ctx.mpc.split_variable(self.x, self.rank_a, self.rank_b)

        xa_encoded_t = self.ctx.mpc.cond_call(lambda: self.encoder.encode(xa).T, lambda: None, dst=self.rank_a)
        xb_encoded_t = self.ctx.mpc.cond_call(lambda: self.encoder.encode(xb).T, lambda: None, dst=self.rank_b)

        # update wa
        # <d.T> @ [xa|rank_a]
        ga = self.ctx.mpc.sshe.smm_mpc_tensor(
            ctx=self.ctx,
            group=self.group,
            op=lambda a, b: b.matmul(a),
            mpc_tensor=dz,
            rank_1=self.rank_a,
            tensor_1=xa_encoded_t,
            rank_2=self.rank_b,
            cipher_2=self.phe_cipher,
        )
        with IgnoreEncodings([ga]):
            ga = ga.div_(self.encoder.scale)

        # <d.T> @ [xb|rank_b]
        gb = self.ctx.mpc.sshe.smm_mpc_tensor(
            ctx=self.ctx,
            group=self.group,
            op=lambda a, b: b.matmul(a),
            mpc_tensor=dz,
            rank_1=self.rank_b,
            tensor_1=xb_encoded_t,
            rank_2=self.rank_a,
            cipher_2=self.phe_cipher,
        )
        with IgnoreEncodings([gb]):
            gb = gb.div_(self.encoder.scale)

        self.wa.grad = ga
        self.wb.grad = gb


class SSHELinearRegressionLossLayer:
    def __init__(self, ctx: Context, rank_a, rank_b, cipher_options=None):
        self.ctx = ctx
        self.group = ctx.mpc.communicator.new_group(
            [rank_a, rank_b], f"{ctx.namespace.federation_tag}.sshe_loss_layer"
        )
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.phe_cipher = ctx.cipher.phe.setup(options=cipher_options)

    def forward(self, z, y):
        dz = z.clone()
        if self.ctx.rank == self.rank_b:
            dz = dz - y
        return SSHESSHELinearRegressionLossLayerLazyLoss(
            self.ctx, self.group, self.rank_a, self.rank_b, self.phe_cipher, dz, z
        )

    def __call__(self, z, y):
        return self.forward(z, y)


class SSHESSHELinearRegressionLossLayerLazyLoss:
    """
    Loss carried out lazily to avoid unnecessary communication
    """

    def __init__(self, ctx, group, rank_a, rank_b, phe_cipher, dz, z):
        self.ctx = ctx
        self.group = group
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.phe_cipher = phe_cipher
        self.dz = dz
        self.z = z

    def get(self, dst=None):
        """
        Computes and returns the loss

        loss = (dz^2).mean()
        """
        dz_mean_square = (
            self.ctx.mpc.sshe.mpc_square(
                ctx=self.ctx,
                group=self.group,
                rank_a=self.rank_a,
                rank_b=self.rank_b,
                x=self.dz,
                cipher_a=self.ctx.mpc.option(self.phe_cipher, self.rank_a),
            )
            .mean()
            .get_plain_text(group=self.group, dst=dst)
        )
        if dst is not None and dst != self.ctx.rank:
            return None
        return dz_mean_square

    def backward(self):
        self.z.backward(self.dz / self.dz.share.shape[0])


class SSHEOptimizerSGD:
    def __init__(self, ctx: Context, params, lr=0.05):
        self.ctx = ctx
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            param -= self.lr * param.grad
            param.grad = None
