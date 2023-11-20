import torch
from fate.arch.context import Context
from fate.arch.utils.trace import auto_trace
from fate.arch.protocol.mpc.common.encoding import IgnoreEncodings
from fate.arch.protocol.mpc.mpc import FixedPointEncoder


class SSHELogisticRegressionLayer:
    def __init__(
        self,
        ctx: Context,
        in_features_a,
        in_features_b,
        out_features,
        rank_a,
        rank_b,
        lr=0.05,
        precision_bits=None,
        generator=None,
    ):
        self.ctx = ctx
        self.group = ctx.mpc.communicator.new_group([rank_a, rank_b], "sshe_aggregator_layer")
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.wa = ctx.mpc.random_tensor(shape=(in_features_a, out_features), src=rank_a, generator=generator)
        self.wb = ctx.mpc.random_tensor(shape=(in_features_b, out_features), src=rank_b, generator=generator)
        self.phe_cipher = ctx.cipher.phe.setup()
        self.precision_bits = precision_bits
        self.lr = lr

    @auto_trace(annotation="[z|rank_b] = [xa|rank_a] * <wa> + [xb|rank_b] * <wb>")
    def forward(self, x):
        xa = x if self.ctx.rank == self.rank_a else None
        xb = x if self.ctx.rank == self.rank_b else None
        s = self.ctx.mpc.sshe.cross_smm(
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
        z = 0.25 * s + 0.5

        # set backward function
        z.backward = SSHELogisticRegressionLayerBackwardFunction(
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


class SSHELogisticRegressionLayerBackwardFunction:
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


class SSHELogisticRegressionLossLayer:
    def __init__(self, ctx: Context, rank_a, rank_b):
        self.ctx = ctx
        self.group = ctx.mpc.communicator.new_group([rank_a, rank_b], "sshe_loss_layer")
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.phe_cipher = ctx.cipher.phe.setup()

    def forward(self, z, y):
        dz = z.clone()
        if self.ctx.rank == self.rank_b:
            dz = dz - y
        return SSHESSHELogisticRegressionLossLayerLazyLoss(
            self.ctx, self.group, self.rank_a, self.rank_b, self.phe_cipher, dz, z
        )

    def __call__(self, z, y):
        return self.forward(z, y)


class SSHESSHELogisticRegressionLossLayerLazyLoss:
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

    def get(self):
        """
        Computes and returns the loss

        loss = 2 * dz.mean()^2 - 0.5 + log(2)
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
            .get_plain_text()
        )
        return 2 * dz_mean_square - 0.5 + torch.log(torch.tensor(2.0))

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
