import typing

import torch

from fate.arch.context import Context
from fate.arch.protocol.mpc.common.rng import generate_random_ring_element_by_seed
from fate.arch.protocol.mpc.primitives.arithmetic import ArithmeticSharedTensor
from fate.arch.protocol.mpc.primitives.beaver import IgnoreEncodings


if typing.TYPE_CHECKING:
    from fate.arch.context import PHECipher


class SSHE:
    @classmethod
    def cross_smm(
        cls,
        ctx: Context,
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

        if ctx.rank == rank_a:
            assert xa is not None, "xa should not be None on rank_a"
            xa = encoder.encode(xa)
            with IgnoreEncodings([wa]):
                z = wa.rmatmul(xa)
            z += cls.smm_lc(ctx, tensor_a=xa, rank_a=rank_a, rank_b=rank_b)
            z += cls.smm_lc(ctx, tensor_b=wb.share, rank_a=rank_b, rank_b=rank_a, cipher=phe_cipher)

        elif ctx.rank == rank_b:
            assert xb is not None, "xb should not be None on rank_b"
            xb = encoder.encode(xb)

            with IgnoreEncodings([wb]):
                z = wb.rmatmul(xb)
            z += cls.smm_lc(ctx, tensor_b=wa.share, rank_a=rank_a, rank_b=rank_b, cipher=phe_cipher)
            z += cls.smm_lc(ctx, tensor_a=xb, rank_a=rank_b, rank_b=rank_a)

        else:
            raise ValueError(f"invalid rank: {ctx.rank}")

        z.encoder = FixedPointEncoder(z.encoder._precision_bits + encoder._precision_bits)
        return z

    @classmethod
    @typing.overload
    def smm_lc(cls, ctx: Context, *, tensor_a, rank_a: int, rank_b: int):
        ...

    @classmethod
    @typing.overload
    def smm_lc(cls, ctx: Context, *, tensor_b, rank_a: int, rank_b: int, cipher: "PHECipher"):
        ...

    @classmethod
    def smm_lc(cls, ctx: Context, **kwargs):
        from fate.arch.context import PHECipher

        tensor_a = kwargs.get("tensor_a", None)
        tensor_b = kwargs.get("tensor_b", None)
        rank_a = kwargs.get("rank_a", None)
        rank_b = kwargs.get("rank_b", None)
        assert (
            isinstance(rank_a, int) and isinstance(rank_b, int) and 0 <= rank_a != rank_b >= 0
        ), "invalid rank_a or rank_b"

        if ctx.rank == rank_b:
            cipher = kwargs.get("cipher", None)
            assert isinstance(cipher, PHECipher), "invalid cipher"
            assert tensor_a is None, "tensor_a should be None on rank_b"
            assert tensor_b is not None, "tensor_b should not be None on rank_a"
            mat_b_enc = cipher.get_tensor_encryptor().encrypt_tensor(tensor_b)
            ctx.mpc.communicator.send_obj(mat_b_enc, rank_a)
            return cls.phe_to_mpc(ctx, src_rank=rank_a, dst_rank=rank_b, phe_cipher=cipher)

        if ctx.rank == rank_a:
            assert tensor_a is not None, "tensor_a should not be None on rank_a"
            assert tensor_b is None, "tensor_b should be None on rank_a"
            mat_b_enc = ctx.mpc.communicator.recv_obj(rank_b)
            enc_z = torch.matmul(tensor_a, mat_b_enc)
            return cls.phe_to_mpc(ctx, src_rank=rank_a, dst_rank=rank_b, phe_tensor=enc_z)

    @classmethod
    @typing.overload
    def smm_rc(cls, *, tensor_a, rank_a: int, rank_b: int, cipher: "PHECipher"):
        ...

    @classmethod
    @typing.overload
    def smm_rc(cls, *, tensor_b, rank_a: int, rank_b: int):
        ...

    @classmethod
    def smm_rc(cls, ctx: Context, **kwargs):
        from fate.arch.context import PHECipher

        tensor_a = kwargs.get("tensor_a", None)
        tensor_b = kwargs.get("tensor_b", None)
        rank_a = kwargs.get("rank_a", None)
        rank_b = kwargs.get("rank_b", None)
        assert (
            isinstance(rank_a, int) and isinstance(rank_b, int) and 0 <= rank_a != rank_b >= 0
        ), "invalid rank_a or rank_b"
        if ctx.rank == rank_a:
            cipher = kwargs.get("cipher", None)
            assert isinstance(cipher, PHECipher), "invalid cipher"
            assert tensor_a is not None, "tensor_a should not be None on rank_a"
            assert tensor_b is None, "tensor_b should be None on rank_a"
            mat_a_enc = cipher.get_tensor_encryptor().encrypt_tensor(tensor_a)
            ctx.mpc.communicator.send_obj(mat_a_enc, rank_b)
            return cls.phe_to_mpc(ctx, src_rank=rank_b, dst_rank=rank_a, phe_cipher=cipher)

        if ctx.rank == rank_b:
            assert tensor_a is None, "tensor_a should be None on rank_b"
            assert tensor_b is not None, "tensor_b should not be None on rank_a"
            mat_a_enc = ctx.mpc.communicator.recv_obj(rank_a)
            enc_z = torch.matmul(mat_a_enc, tensor_b)
            return cls.phe_to_mpc(ctx, src_rank=rank_b, dst_rank=rank_a, phe_tensor=enc_z)

    @classmethod
    @typing.overload
    def phe_to_mpc(cls, ctx: Context, src_rank, dst_rank, phe_cipher):
        ...

    @classmethod
    @typing.overload
    def phe_to_mpc(cls, ctx: Context, phe_tensor, src_rank, dst_rank):
        ...

    @classmethod
    def phe_to_mpc(cls, ctx: Context, **kwargs):
        """
        Convert a phe-tensor to MPC encrypted tensor.
        """
        src_rank = kwargs.get("src_rank", None)
        assert isinstance(src_rank, int) and 0 <= src_rank < ctx.world_size, f"invalid src_rank: {src_rank}"
        dst_rank = kwargs.get("dst_rank", None)
        assert isinstance(dst_rank, int) and 0 <= dst_rank < ctx.world_size, f"invalid dst_rank: {dst_rank}"
        phe_tensor = kwargs.get("phe_tensor", None)
        phe_cipher = kwargs.get("phe_cipher", None)
        if ctx.rank == src_rank:
            assert phe_tensor is not None, "he_tensor should not be None on src_rank"
            assert phe_cipher is None, "phe_cipher should be None on src_rank"

            src_share = generate_random_ring_element_by_seed(phe_tensor.shape, None)
            dst_share = phe_tensor - src_share
            ctx.mpc.communicator.send(dst_share, dst=dst_rank)
            return ArithmeticSharedTensor.from_shares(ctx, src_share)

        else:
            assert phe_tensor is None, "he_tensor should be None on dst_rank"
            assert phe_cipher is not None, "phe_cipher should not be None on dst_rank"
            dst_share = ctx.mpc.communicator.recv(None, src=src_rank)
            dst_share = phe_cipher.get_tensor_decryptor().decrypt_tensor(dst_share)
            return ArithmeticSharedTensor.from_shares(ctx, dst_share)

    @classmethod
    @typing.overload
    def smm_sa_rb_lc(cls, ctx: Context, sa: ArithmeticSharedTensor, rank_a, rank_b, *, tensor_b):
        ...

    @classmethod
    @typing.overload
    def smm_sa_rb_lc(cls, ctx: Context, sa: ArithmeticSharedTensor, rank_a, rank_b, *, phe_cipher):
        ...

    @classmethod
    def smm_sa_rb_lc(cls, ctx: Context, sa: ArithmeticSharedTensor, rank_a, rank_b, **kwargs):
        """
        Securely computes c = sa.T @ b, where:
            1. sa is a shared tensor.
            2. b is a tensor located on rank_b.
            3. c is a tensor located on rank_a.
        """
        from fate.arch.context import PHECipher

        if ctx.rank == rank_a:
            phe_cipher = kwargs.get("phe_cipher", None)
            assert isinstance(phe_cipher, PHECipher), "invalid phe_cipher"
            enc_sa_share = phe_cipher.get_tensor_encryptor().encrypt_tensor(sa.share.T)
            ctx.mpc.communicator.send(enc_sa_share, rank_b)
            enc_c = ctx.mpc.communicator.recv(None, rank_b)
            c = phe_cipher.get_tensor_decryptor().decrypt_tensor(enc_c)
            return c

        if ctx.rank == rank_b:
            tensor_b = kwargs.get("tensor_b", None)
            assert isinstance(tensor_b, torch.Tensor), "invalid tensor_b"
            enc_sa_share = ctx.mpc.communicator.recv(None, rank_a)
            enc_sa_share += sa.share.T
            enc_c = tensor_b @ enc_sa_share
            ctx.mpc.communicator.send(enc_c, rank_a)
