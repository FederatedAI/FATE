import logging


import torch

from . import MPCModule
from ...arch import Context
from ...arch.tensor import DTensor
from .mpc_sa_layer import SSHEAggregatorLayer

logger = logging.getLogger(__name__)


class Toy(MPCModule):
    def __init__(
        self,
    ):
        ...

    def fit(self, ctx: Context) -> None:
        self.fit_matmul(ctx)
        # x = _get_left_tensor(ctx, 0)
        # alice = ctx.mpc.cond_call(lambda: x, lambda: _get_left_tensor(ctx, is_zero=True), dst=0)
        # logger.info(torch.to_local_f(x))
        # logger.info(f"expect={torch.to_local_f(torch.log(x))}")
        # alice_enc = ctx.mpc.encrypt(alice)
        # out = alice_enc.log().get_plain_text()
        # logger.info(f"exp={torch.to_local_f(out)}")

    def sshe_he_to_mpc(self, ctx: Context):
        phe = ctx.cipher.phe.broadcast(src=1)
        if ctx.rank == 0:
            x = torch.randint(-100, 100, (2, 3))
            logger.info(f"plain={torch.to_local_f(x)}")
            enc_x = phe.get_tensor_encryptor().encrypt_tensor(x)
            xs = ctx.mpc.sshe_he_to_mpc(phe_tensor=enc_x)
        else:
            xs = ctx.mpc.sshe_he_to_mpc(decryptor=phe.get_tensor_decryptor())
        logger.info(xs.reveal())

    def fit_mul(self, ctx: Context):
        x = _get_left_tensor(ctx, 0)
        y = _get_left_tensor(ctx, 1)
        expect = torch.mul(x, y)
        logger.info(f"expect={torch.to_local_f(expect)}")

        x_alice = ctx.mpc.cond_call(lambda: x, lambda: _get_left_tensor(ctx, is_zero=True), dst=0)
        x_alice_enc = ctx.mpc.encrypt(x_alice, src=0)

        x_bob = ctx.mpc.cond_call(lambda: y, lambda: _get_left_tensor(ctx, is_zero=True), dst=1)
        x_bob_enc = ctx.mpc.encrypt(x_bob, src=1)

        out = x_bob_enc.mul(x_alice_enc).get_plain_text()
        ctx.mpc.info(f"mul={torch.to_local_f(out)}")

    def fit_matmul(self, ctx: Context):

        with ctx.mpc.communicator.new_group(ranks=[0,1], name="matmul"):
            x = _get_left_tensor(ctx, 0)
            y = _get_right_tensor(ctx, 1)
            expect = torch.matmul(x, y)
            logger.info(f"expect={torch.to_local_f(expect)}")

            x_alice = ctx.mpc.cond_call(
                lambda: _get_left_tensor(ctx, 0), lambda: _get_left_tensor(ctx, is_zero=True), dst=0
            )
            ctx.mpc.info(torch.to_local_f(x_alice), dst=0)
            x_alice_enc = ctx.mpc.encrypt(x_alice, src=0)

            x_bob = ctx.mpc.cond_call(
                lambda: _get_right_tensor(ctx, 1), lambda: _get_right_tensor(ctx, is_zero=True), dst=1
            )
            ctx.mpc.info(torch.to_local_f(x_bob), dst=1)
            x_bob_enc = ctx.mpc.encrypt(x_bob, src=1)

            out = x_alice_enc.matmul(x_bob_enc).get_plain_text()
            ctx.mpc.info(f"matmul={torch.to_local_f(out)}")


def _get_left_tensor(ctx, seed=None, is_zero=False):
    if is_zero:
        return zero(ctx, _shapes1(), axis=0)
    else:
        return rand(ctx, _shapes1(), axis=0, seed=seed)
    # return rand(ctx, _shapes2(), axis=1, seed=seed)


def _get_right_tensor(ctx, seed=None, is_zero=False):
    if is_zero:
        return torch.zeros(2, 10)
    else:
        generator = torch.Generator()
        if seed:
            generator.manual_seed(seed)
        return torch.rand(2, 10, generator=generator)
    # return rand(ctx, _shapes1(), axis=0, seed=seed)


def _shapes1():
    return [(2, 2), (4, 2)]


def _shapes2():
    return [(3, 2), (3, 4)]


def rand(ctx, shapes, axis=0, seed=None):
    generator = torch.Generator()
    if seed:
        generator.manual_seed(seed)
    return DTensor.from_sharding_list(ctx, [torch.rand(shape, generator=generator) for shape in shapes], axis=axis)


def zero(ctx, shapes, axis=0):
    return DTensor.from_sharding_list(ctx, [torch.zeros(shape) for shape in shapes], axis=axis)
