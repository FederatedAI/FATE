from typing import List, Optional, Tuple

import torch
from fate_utils.histogram import PK as _PK
from fate_utils.histogram import SK as _SK
from fate_utils.histogram import Coders as _Coder
from fate_utils.histogram import FixedpointPaillierVector, FixedpointVector
from fate_utils.histogram import keygen as _keygen


class SK:
    def __init__(self, sk: _SK):
        self.sk = sk

    def decrypt_vec(self, vec: FixedpointPaillierVector) -> FixedpointVector:
        return self.sk.decrypt_vec(vec)


class PK:
    def __init__(self, pk: _PK):
        self.pk = pk

    def encrypt_vec(self, vec, obfuscate) -> FixedpointPaillierVector:
        return self.pk.encrypt_vec(vec, obfuscate)

    def encrypt(self, val, obfuscate) -> FixedpointPaillierVector:
        return self.pk.encrypt(val, obfuscate)


class Coder:
    def __init__(self, coder: _Coder):
        self.coder = coder

    def encode_vec(self, vec: torch.Tensor, dtype: torch.dtype = None) -> FixedpointVector:
        if dtype is None:
            dtype = vec.dtype
        if dtype == torch.float64:
            return self.encode_f64_vec(vec)
        if dtype == torch.float32:
            return self.encode_f32_vec(vec)
        if dtype == torch.int64:
            return self.encode_i64_vec(vec)
        if dtype == torch.int32:
            return self.encode_i32_vec(vec)
        raise NotImplementedError(f"{vec.dtype} not supported")

    def decode_vec(self, vec: FixedpointVector, dtype: torch.dtype) -> torch.Tensor:
        if dtype == torch.float64:
            return self.decode_f64_vec(vec)
        if dtype == torch.float32:
            return self.decode_f32_vec(vec)
        if dtype == torch.int64:
            return self.decode_i64_vec(vec)
        if dtype == torch.int32:
            return self.decode_i32_vec(vec)
        raise NotImplementedError(f"{dtype} not supported")

    def encode(self, val, dtype=None) -> FixedpointVector:
        if isinstance(val, torch.Tensor):
            assert val.ndim == 0, "only scalar supported"
            if dtype is None:
                dtype = val.dtype
            val = val.item()
        if dtype == torch.float64:
            return self.encode_f64(val)
        if dtype == torch.float32:
            return self.encode_f32(val)
        if dtype == torch.int64:
            return self.encode_i64(val)
        if dtype == torch.int32:
            return self.encode_i32(val)
        raise NotImplementedError(f"{dtype} not supported")

    def encode_f64(self, val: float):
        return self.coder.encode_f64(val)

    def decode_f64(self, val):
        return self.coder.decode_f64(val)

    def encode_i64(self, val: int):
        return self.coder.encode_i64(val)

    def decode_i64(self, val):
        return self.coder.decode_i64(val)

    def encode_f32(self, val: float):
        return self.coder.encode_f32(val)

    def decode_f32(self, val):
        return self.coder.decode_f32(val)

    def encode_i32(self, val: int):
        return self.coder.encode_i32(val)

    def decode_i32(self, val):
        return self.coder.decode_i32(val)

    def encode_f64_vec(self, vec: torch.Tensor):
        return self.coder.encode_f64_vec(vec.detach().numpy())

    def decode_f64_vec(self, vec):
        return torch.tensor(self.coder.decode_f64_vec(vec))

    def encode_i64_vec(self, vec: torch.Tensor):
        return self.coder.encode_i64_vec(vec.detach().numpy())

    def decode_i64_vec(self, vec):
        return torch.tensor(self.coder.decode_i64_vec(vec))

    def encode_f32_vec(self, vec: torch.Tensor):
        return self.coder.encode_f32_vec(vec.detach().numpy())

    def decode_f32_vec(self, vec):
        return torch.tensor(self.coder.decode_f32_vec(vec))

    def encode_i32_vec(self, vec: torch.Tensor):
        return self.coder.encode_i32_vec(vec.detach().numpy())

    def decode_i32_vec(self, vec):
        return torch.tensor(self.coder.decode_i32_vec(vec))


def keygen(key_size):
    sk, pk, coder = _keygen(key_size)
    return SK(sk), PK(pk), Coder(coder)


class ops:
    @staticmethod
    def add(a, b, pk: PK):
        return a.add(pk.pk, b)

    @staticmethod
    def add_vec(a, b: torch.Tensor, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_vec(b, dtype=output_dtype)
        encrypted = pk.encrypt_vec(encoded, obfuscate=False)
        return a.add(pk.pk, encrypted)

    @staticmethod
    def add_scalar(a, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt(encoded, obfuscate=False)
        return a.add_scalar(pk.pk, encrypted)

    @staticmethod
    def rsub(a, b, pk: PK):
        return a.rsub(pk.pk, b)

    @staticmethod
    def rsub_vec(a, b: torch.Tensor, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_vec(b, dtype=output_dtype)
        encrypted = pk.encrypt_vec(encoded, obfuscate=False)
        return a.rsub(pk.pk, encrypted)

    @staticmethod
    def rsub_scalar(a, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt(encoded, obfuscate=False)
        return a.rsub_scalar(pk.pk, encrypted)

    @staticmethod
    def sub(a, b, pk: PK):
        return a.sub(pk.pk, b)

    @staticmethod
    def sub_vec(a, b: torch.Tensor, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_vec(b, dtype=output_dtype)
        encrypted = pk.encrypt_vec(encoded, obfuscate=False)
        return a.sub(pk.pk, encrypted)

    @staticmethod
    def sub_scalar(a, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt(encoded, obfuscate=False)
        return a.sub_scalar(pk.pk, encrypted)

    @staticmethod
    def mul_vec(a, b: torch.Tensor, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_vec(b, dtype=output_dtype)
        return a.mul(pk.pk, encoded)

    @staticmethod
    def mul_scalar(a, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        return a.mul_scalar(pk.pk, encoded)

    @staticmethod
    def matmul(a, b: torch.Tensor, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode_vec(b, dtype=output_dtype)
        # TODO: move this to python side so other protocols can use it without matmul support?
        return a.matmul(pk.pk, encoded, a_shape, b_shape)

    @staticmethod
    def rmatmul(a, b: torch.Tensor, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode_vec(b, dtype=output_dtype)
        return a.rmatmul(pk.pk, encoded, a_shape, b_shape)

    @staticmethod
    def zeros(size):
        return FixedpointPaillierVector.zeros(size)

    @staticmethod
    def i_add_vec(pk: PK, a, b, sa=0, sb=0, size: Optional[int] = None):
        if a is b:
            a.iadd_vec_self(sa, sb, size, pk.pk)
        else:
            a.iadd_vec(b, sa, sb, size, pk.pk)

    @staticmethod
    def slice(a, start, size):
        return a.slice(start, size)

    @staticmethod
    def intervals_sum_with_step(pk: PK, a, intervals: List[Tuple[int, int]], step: int):
        """
        sum in the given intervals, with step size

        for example:
            if step=2, intervals=[(0, 4), (6, 12)], a = [a0, a1, a2, a3, a4, a5, a6, a7,...]
            then the result is [a0+a2, a1+a3, a6+a8+a10, a7+a9+a11]
        """
        return a.intervals_sum_with_step(pk.pk, intervals, step)

    @staticmethod
    def chunking_cumsum_with_step(pk: PK, a, chunk_sizes: List[int], step: int):
        """
        chunking cumsum with step size

        for example:
            if step=2, chunk_sizes=[4, 2, 6], a = [a0, a1, a2, a3, a4, a5, a6, a7,...a11]
            then the result is [a0, a1, a0+a2, a1+a3, a4, a5, a6, a7, a6+a8, a7+a9, a6+a8+a10, a7+a9+a11]
        Args:
            pk: the public key
            a: the vector to cumsum
            chunk_sizes: the chunk sizes, must sum to a.size
            step: the step size, cumsum with skip step-1 elements
        Returns:
            the cumsum result
        """
        return a.chunking_cumsum_with_step(pk.pk, chunk_sizes, step)
