import typing

import torch

from fate.arch.context import Context
from fate.arch.tensor import DTensor
from fate.arch.protocol.mpc.common.rng import generate_random_ring_element
from fate.arch.protocol.mpc.primitives.arithmetic import ArithmeticSharedTensor
from fate.arch.protocol.mpc.primitives.beaver import IgnoreEncodings
from fate.arch.trace import auto_trace


if typing.TYPE_CHECKING:
    from fate.arch.context import PHECipher


class SSHE:
    @classmethod
    @auto_trace(annotation="<z> = [xa|rank_a] * <wa> + [xb|rank_b] * <wb>")
    def cross_smm(
        cls,
        ctx: Context,
        group,
        xa,
        xb,
        wa: ArithmeticSharedTensor,
        wb: ArithmeticSharedTensor,
        rank_a,
        rank_b,
        phe_cipher,
        precision_bits=None,
    ):
        """
        Securely computes xa * wa + xb * wb, where:
            1. xa is a tensor that belongs to rank_a, and xb is a tensor that belongs to rank_b.
            2. wa is a shared tensor for rank_a and rank_b.
            3. wb is a shared tensor for rank_a and rank_b.

            Input:    xa   wa.share_a   wb.share_a       xb   wa.share_b   wb.share_b

            Comp:       (xa * wa.share_b).share_a          (xa * wa.share_b).share_b         (SMM)
                        (xb * wa.share_a).share_a          (xb * wa.share_a).share_b         (SMM)
                            xa * wa.share_a                       xb * wb.share_b

            Output:  z.share_a = (xa * wa.share_b).share_a + (xb * wb.share_a).share_a + (xa * wb.share_a)
                     z.share_b = (xa * wa.share_b).share_b + (xb * wb.share_a).share_b + (xb * wa.share_b)
        """
        from fate.arch.protocol.mpc.mpc import FixedPointEncoder
        from fate.arch.context import PHECipher

        assert isinstance(wa, ArithmeticSharedTensor), "invalid wa"
        assert isinstance(wb, ArithmeticSharedTensor), "invalid wb"
        assert isinstance(rank_a, int) and 0 <= rank_a < ctx.world_size, f"invalid rank_a: {rank_a}"
        assert isinstance(rank_b, int) and 0 <= rank_b < ctx.world_size, f"invalid rank_b: {rank_b}"
        assert isinstance(phe_cipher, PHECipher), "invalid phe_cipher"
        encoder = FixedPointEncoder(precision_bits)

        x = ctx.mpc.cond_call(lambda: xa, lambda: xb, dst=rank_a)
        encoded_x = encoder.encode(x)
        w = ctx.mpc.cond_call(lambda: wa, lambda: wb, dst=rank_a)
        with IgnoreEncodings([w]):
            z = w.rmatmul(encoded_x)

        # [xa|rank_a] @ [wa.share|rank_b]
        z += cls.smm(
            ctx,
            group,
            op=lambda a, b: torch.matmul(b, a),
            rank_1=rank_b,
            tensor_1=ctx.mpc.option(wa.share, rank_b),
            cipher_1=ctx.mpc.option(phe_cipher, rank_b),
            rank_2=rank_a,
            tensor_2=ctx.mpc.option(encoded_x, rank_a),
        )
        # [xb|rank_b] @ [wb.share|rank_a]
        z += cls.smm(
            ctx,
            group,
            op=lambda a, b: torch.matmul(b, a),
            rank_1=rank_a,
            tensor_1=ctx.mpc.option(wb.share, rank_a),
            cipher_1=ctx.mpc.option(phe_cipher, rank_a),
            rank_2=rank_b,
            tensor_2=ctx.mpc.option(encoded_x, rank_b),
        )
        with IgnoreEncodings([z]):
            z.div_(encoder.scale)
        return z

    @classmethod
    @auto_trace
    def smm(
        cls,
        ctx: Context,
        group,
        op: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rank_1,
        tensor_1,
        cipher_1,
        rank_2,
        tensor_2,
    ) -> ArithmeticSharedTensor:
        """
        Securely computes tensor_3 = op(tensor_1, tensor_2), where:
            1. tensor_1 is a tensor that belongs to rank_1
            2. tensor_2 is a tensor that belongs to rank_2.
            3. tensor_3 is a mpc tensor that belongs to rank_1 and rank_2.

        the cipher is provided by rank_1 and is used to encrypt tensor_1.
        One should choose the smaller tensor to be encrypted by PHE to improve performance.
        """
        from fate.arch.context import PHECipher

        assert (
            isinstance(rank_1, int) and isinstance(rank_2, int) and 0 <= rank_1 != rank_2 >= 0
        ), "invalid rank_a or rank_b"
        ctx.mpc.option_assert(lambda: tensor_1 is not None, "tensor_1 should not be None on rank_1", dst=rank_1)
        ctx.mpc.option_assert(lambda: tensor_1 is None, "tensor_1 should not be None on rank_2", dst=rank_2)
        ctx.mpc.option_assert(lambda: tensor_2 is not None, "tensor_2 should not be None on rank_2", dst=rank_2)
        ctx.mpc.option_assert(lambda: tensor_2 is None, "tensor_2 should be None on rank_1", dst=rank_1)
        ctx.mpc.option_assert(lambda: isinstance(cipher_1, PHECipher), "invalid cipher", dst=rank_1)

        tensor_1_enc = ctx.mpc.communicator.broadcast_obj(
            obj=ctx.mpc.option_call(lambda: cipher_1.get_tensor_encryptor().encrypt_tensor(tensor_1), dst=rank_1),
            src=rank_1,
            group=group,
        )
        return cls.phe_to_mpc(
            ctx,
            group,
            rank_a=rank_2,
            rank_b=rank_1,
            phe_cipher_b=ctx.mpc.option(cipher_1, dst=rank_1),
            phe_tensor_a=ctx.mpc.option_call(lambda: op(tensor_1_enc, tensor_2), dst=rank_2),
        )

    @classmethod
    def smm_mpc_tensor(
        cls,
        ctx: Context,
        group,
        *,
        op: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mpc_tensor: ArithmeticSharedTensor,
        rank_1: int,
        tensor_1: torch.Tensor,
        rank_2: int,
        cipher_2: "PHECipher",
    ):
        """
        Securely computes c = op(mpc_tensor, tensor_1), where:
            1. mpc_tensor is a shared tensor that belongs to rank_a and rank_b.
            2. tensor_1 is a tensor that belongs to rank_1.
            3. c is a shared tensor that belongs to rank_a and rank_b.

        """
        ga = cls.smm(
            ctx,
            group,
            op=lambda a, b: op(a, b),
            rank_1=rank_2,
            tensor_1=ctx.mpc.cond_call(lambda: mpc_tensor.share, lambda: None, rank_2),
            cipher_1=ctx.mpc.option(cipher_2, dst=rank_2),
            rank_2=rank_1,
            tensor_2=ctx.mpc.option(tensor_1, rank_1),
        )
        if ctx.rank == rank_1:
            ga.share += op(mpc_tensor.share, tensor_1)
        return ga

    @classmethod
    @auto_trace
    def phe_to_mpc(cls, ctx: Context, group, rank_a, rank_b, phe_tensor_a, phe_cipher_b):
        """
        Convert a phe-tensor to MPC encrypted tensor.

        phe_tensor_a: a phe-tensor that belongs to rank_a with corresponding phe_cipher_b holding by rank_b.
        """
        ctx.mpc.option_assert(
            lambda: phe_tensor_a is not None, "phe_tensor should not be None on src_rank", dst=rank_a
        )
        ctx.mpc.option_assert(lambda: phe_cipher_b is None, "phe_cipher should be None on src_rank", dst=rank_a)
        ctx.mpc.option_assert(lambda: phe_tensor_a is None, "phe_tensor should be None on dst_rank", dst=rank_b)
        ctx.mpc.option_assert(
            lambda: phe_cipher_b is not None, "phe_cipher should not be None on dst_rank", dst=rank_b
        )
        assert isinstance(rank_a, int) and 0 <= rank_a < ctx.world_size, f"invalid src_rank: {rank_a}"
        assert isinstance(rank_b, int) and 0 <= rank_b < ctx.world_size, f"invalid dst_rank: {rank_b}"
        src_share = ctx.mpc.option_call(
            lambda: generate_random_ring_element(
                ctx, phe_tensor_a.shardings.shapes if isinstance(phe_tensor_a, DTensor) else phe_tensor_a.shape
            ),
            dst=rank_a,
        )
        dst_share = ctx.mpc.option_call(lambda: phe_tensor_a - src_share, dst=rank_a)
        dst_share = ctx.mpc.communicator.broadcast(dst_share, src=rank_a, group=group)
        dst_share = ctx.mpc.option_call(
            lambda: phe_cipher_b.get_tensor_decryptor().decrypt_tensor(dst_share), dst=rank_b
        )
        share = ctx.mpc.cond(src_share, dst_share, dst=rank_a)
        return ArithmeticSharedTensor.from_shares(ctx, share)

    @classmethod
    @auto_trace
    def mpc_to_he(cls, ctx: Context, src_rank, dst_rank, phe_tensor=None, phe_cipher=None):
        ...

    @classmethod
    @auto_trace
    def mpc_square(cls, ctx: Context, group, rank_a, rank_b, x: ArithmeticSharedTensor, cipher_a):
        """
        Securely computes x * x, where:
            1. x is a shared tensor that belongs to rank_a and rank_b.
            2. cipher_a is a PHE cipher that belongs to rank_a.
        """
        x_a_enc = ctx.mpc.communicator.broadcast_obj(
            obj=ctx.mpc.option_call(lambda: cipher_a.get_tensor_encryptor().encrypt_tensor(x.share), dst=rank_a),
            src=rank_a,
        )
        x_ab = cls.phe_to_mpc(
            ctx,
            group,
            rank_a=rank_b,
            rank_b=rank_a,
            phe_cipher_b=ctx.mpc.option(cipher_a, dst=rank_a),
            phe_tensor_a=ctx.mpc.option_call(lambda: x_a_enc * x.share, dst=rank_b),
        )
        x_ab.share = x.share * x.share + x_ab.share * 2
        with IgnoreEncodings([x_ab]):
            x_ab.div_(x.encoder.scale)
        return x_ab
