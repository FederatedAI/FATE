import typing
from typing import Any, Iterator

import torch
from torch.nn import Parameter

from fate.arch.context import Context
from fate.arch.protocol.mpc.common.encoding import IgnoreEncodings
from fate.arch.protocol.mpc.mpc import FixedPointEncoder
from fate.arch.trace import auto_trace


class SSHENeuralNetworkAggregatorLayer(torch.nn.Module):
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
    ):
        self.group = ctx.mpc.communicator.new_group(
            [rank_a, rank_b], f"{ctx.namespace.federation_tag}.sshe_nn_aggregator_layer"
        )
        self.aggregator = SSHENeuralNetworkAggregator(
            ctx,
            in_features_a=in_features_a,
            in_features_b=in_features_b,
            out_features=out_features,
            rank_a=rank_a,
            rank_b=rank_b,
            group=self.group,
            encoder=FixedPointEncoder(precision_bits),
            wa_init_fn=wa_init_fn,
            wb_init_fn=wb_init_fn,
        )
        super().__init__()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.aggregator.parameters()

    def forward(self, input):
        return SSHENeuralNetworkAggregatorFunction.apply(input, self.aggregator)

    def get_wa(self, dst=None):
        return self.aggregator.wa.get_plain_text(dst=dst, group=self.group)

    def get_wb(self, dst=None):
        return self.aggregator.wb.get_plain_text(dst=dst, group=self.group)


class SSHENeuralNetworkAggregatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, aggregator: "SSHENeuralNetworkAggregator"):
        ctx.save_for_backward(input)
        output = aggregator.forward(input)
        output = output.get_plain_text(dst=aggregator.rank_b)
        ctx.aggregator = aggregator
        return aggregator.ctx.mpc.cond_call(lambda: output, lambda: torch.empty(1), dst=aggregator.rank_b)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        aggregator: "SSHENeuralNetworkAggregator" = ctx.aggregator
        ha = ctx.saved_tensors[0] if aggregator.ctx.rank == aggregator.rank_a else None
        hb = ctx.saved_tensors[0] if aggregator.ctx.rank == aggregator.rank_b else None
        dz = grad_outputs[0] if aggregator.ctx.rank == aggregator.rank_b else None
        return aggregator.backward(dz, ha, hb), None


class SSHENeuralNetworkAggregator:
    def __init__(
        self,
        ctx: Context,
        in_features_a,
        in_features_b,
        out_features,
        rank_a,
        rank_b,
        group,
        encoder,
        wa_init_fn,
        wb_init_fn,
        precision_bits=None,
        cipher_options=None,
    ):
        self.ctx = ctx
        self.wa = ctx.mpc.init_tensor(shape=(in_features_a, out_features), init_func=wa_init_fn, src=rank_a)
        self.wb = ctx.mpc.init_tensor(shape=(in_features_b, out_features), init_func=wb_init_fn, src=rank_b)
        self.phe_cipher = ctx.cipher.phe.setup(options=cipher_options)
        self.rank_a = rank_a
        self.rank_b = rank_b
        self.group = group
        self.precision_bits = precision_bits
        self.encoder = encoder

    def parameters(self) -> Iterator[Parameter]:
        yield self.wa
        yield self.wb

    @auto_trace(annotation="[z|rank_b] = [xa|rank_a] * <wa> + [xb|rank_b] * <wb>")
    def forward(self, input):
        xa, xb = self.ctx.mpc.split_variable(input, self.rank_a, self.rank_b)
        out = self.ctx.mpc.sshe.cross_smm(
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
        return out

    @auto_trace
    def backward(self, dz, ha, hb):
        ha_encoded_t = self.ctx.mpc.cond_call(lambda: self.encoder.encode(ha).T, lambda: None, dst=self.rank_a)
        dz_encoded = self.ctx.mpc.cond_call(lambda: self.encoder.encode(dz), lambda: None, dst=self.rank_b)

        # dh
        enc_wa_share = self.ctx.mpc.option_call(
            lambda: self.phe_cipher.get_tensor_encryptor().encrypt_tensor(self.wa.share.T), dst=self.rank_a
        )
        enc_wa_share = self.ctx.mpc.communicator.broadcast(enc_wa_share, src=self.rank_a, group=self.group)
        enc_dha = self.ctx.mpc.option_call(lambda: dz_encoded @ (enc_wa_share + self.wa.share.T), dst=self.rank_b)
        enc_dha = self.ctx.mpc.communicator.broadcast(enc_dha, src=self.rank_b, group=self.group)
        dha = self.ctx.mpc.option_call(
            lambda: self.phe_cipher.get_tensor_decryptor().decrypt_tensor(enc_dha) / self.wa.encoder.scale**2,
            dst=self.rank_a,
        )
        enc_wb_share = self.ctx.mpc.option_call(
            lambda: self.phe_cipher.get_tensor_encryptor().encrypt_tensor(self.wb.share.T), self.rank_a
        )
        enc_wb_share = self.ctx.mpc.communicator.broadcast(enc_wb_share, src=self.rank_a, group=self.group)
        enc_dhb = self.ctx.mpc.option_call(lambda: dz_encoded @ enc_wb_share, dst=self.rank_b)
        enc_dhb = self.ctx.mpc.communicator.broadcast(enc_dhb, src=self.rank_b, group=self.group)
        dhb = self.ctx.mpc.option_call(
            lambda: self.phe_cipher.get_tensor_decryptor().decrypt_tensor(enc_dhb), dst=self.rank_a
        )
        dhb = self.ctx.mpc.communicator.broadcast(dhb, src=self.rank_a, group=self.group)
        dhb = self.ctx.mpc.option_call(
            lambda: (dhb + dz_encoded @ self.wb.share.T) / self.wb.encoder.scale**2, dst=self.rank_b
        )
        dh = self.ctx.mpc.cond(dha, dhb, dst=self.rank_a)

        # update wa
        ga = self.ctx.mpc.sshe.smm(
            self.ctx,
            group=self.group,
            op=torch.matmul,
            rank_1=self.rank_a,
            tensor_1=ha_encoded_t,
            cipher_1=self.phe_cipher,
            rank_2=self.rank_b,
            tensor_2=dz_encoded,
        )
        ga = _rescaler(ga, self.encoder)

        # update wb
        gb = self.ctx.mpc.option_call(lambda: hb.T @ dz, dst=self.rank_b)

        # set grad for wa and wb
        _set_grad(self.wa, ga)
        _set_grad(self.wb, gb)

        return dh


def _rescaler(x, encoder):
    with IgnoreEncodings([x]):
        x = x.div_(encoder.scale)
    return x


def _set_grad(x, grad):
    if getattr(x, "grad", None) is None:
        x.grad = grad
    else:
        x.grad += grad


class SSHENeuralNetworkOptimizerSGD:
    def __init__(self, ctx: Context, params, lr):
        self.ctx = ctx
        self.params = list(params)
        self.lr = lr

    @auto_trace(annotation="<param> -= <lr> * <param.grad>")
    def step(self):
        for param in self.params:
            if getattr(param, "grad", None) is None:
                continue
            param -= self.lr * param.grad
            param.grad = None
