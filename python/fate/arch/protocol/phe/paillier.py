from typing import List, Optional, Tuple

import torch
from fate_utils.histogram import PK as _PK
from fate_utils.histogram import SK as _SK
from fate_utils.histogram import Coders as _Coder
from fate_utils.histogram import FixedpointPaillierVector, FixedpointVector
from fate_utils.histogram import keygen as _keygen

from .type import TensorEvaluator

V = torch.Tensor
EV = FixedpointPaillierVector
FV = FixedpointVector


class SK:
    def __init__(self, sk: _SK):
        self.sk = sk

    def decrypt_to_encoded(self, vec: EV) -> FV:
        return self.sk.decrypt_to_encoded(vec)


class PK:
    def __init__(self, pk: _PK):
        self.pk = pk

    def encrypt_encoded(self, vec: FV, obfuscate: bool) -> EV:
        return self.pk.encrypt_encoded(vec, obfuscate)

    def encrypt_encoded_scalar(self, val, obfuscate) -> FixedpointPaillierVector:
        return self.pk.encrypt_encoded_scalar(val, obfuscate)


class Coder:
    def __init__(self, coder: _Coder):
        self.coder = coder

    def encode_tensor(self, tensor: V, dtype: torch.dtype = None) -> FV:
        if dtype is None:
            dtype = tensor.dtype
        return self.encode_vec(tensor.flatten(), dtype=dtype)

    def decode_tensor(self, tensor: FV, dtype: torch.dtype, shape: torch.Size = None) -> V:
        data = self.decode_vec(tensor, dtype)
        if shape is not None:
            data = data.reshape(shape)
        return data

    def encode_vec(self, vec: V, dtype: torch.dtype = None) -> FV:
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

    def decode_vec(self, vec: FV, dtype: torch.dtype) -> V:
        if dtype == torch.float64:
            return self.decode_f64_vec(vec)
        if dtype == torch.float32:
            return self.decode_f32_vec(vec)
        if dtype == torch.int64:
            return self.decode_i64_vec(vec)
        if dtype == torch.int32:
            return self.decode_i32_vec(vec)
        raise NotImplementedError(f"{dtype} not supported")

    def encode(self, val, dtype=None) -> FV:
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
        vec = vec.detach().flatten()
        return self.coder.encode_f64_vec(vec.detach().numpy())

    def decode_f64_vec(self, vec):
        return torch.tensor(self.coder.decode_f64_vec(vec))

    def encode_i64_vec(self, vec: torch.Tensor):
        vec = vec.detach().flatten()
        return self.coder.encode_i64_vec(vec.detach().numpy())

    def decode_i64_vec(self, vec):
        return torch.tensor(self.coder.decode_i64_vec(vec))

    def encode_f32_vec(self, vec: torch.Tensor):
        vec = vec.detach().flatten()
        return self.coder.encode_f32_vec(vec.detach().numpy())

    def decode_f32_vec(self, vec):
        return torch.tensor(self.coder.decode_f32_vec(vec))

    def encode_i32_vec(self, vec: torch.Tensor):
        vec = vec.detach().flatten()
        return self.coder.encode_i32_vec(vec.detach().numpy())

    def decode_i32_vec(self, vec):
        return torch.tensor(self.coder.decode_i32_vec(vec))


def keygen(key_size):
    sk, pk, coder = _keygen(key_size)
    return SK(sk), PK(pk), Coder(coder)


class evaluator(TensorEvaluator[EV, V, PK, Coder]):
    @staticmethod
    def add(a: EV, b: EV, pk: PK):
        return a.add(pk.pk, b)

    @staticmethod
    def add_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded(encoded, obfuscate=False)
        return a.add(pk.pk, encrypted)

    @staticmethod
    def add_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return a.add_scalar(pk.pk, encrypted)

    @staticmethod
    def sub(a: EV, b: EV, pk: PK):
        return a.sub(pk.pk, b)

    @staticmethod
    def sub_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded(encoded, obfuscate=False)
        return a.sub(pk.pk, encrypted)

    @staticmethod
    def sub_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return a.sub_scalar(pk.pk, encrypted)

    @staticmethod
    def rsub(a: EV, b: EV, pk: PK):
        return a.rsub(pk.pk, b)

    @staticmethod
    def rsub_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded(encoded, obfuscate=False)
        return a.rsub(pk.pk, encrypted)

    @staticmethod
    def rsub_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return a.rsub_scalar(pk.pk, encrypted)

    @staticmethod
    def mul_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        return a.mul(pk.pk, encoded)

    @staticmethod
    def mul_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        return a.mul_scalar(pk.pk, encoded)

    @staticmethod
    def matmul(a: EV, b: V, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        # TODO: move this to python side so other protocols can use it without matmul support?
        return a.matmul(pk.pk, encoded, a_shape, b_shape)

    @staticmethod
    def rmatmul(a: EV, b: V, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        return a.rmatmul(pk.pk, encoded, a_shape, b_shape)

    @staticmethod
    def zeros(size) -> EV:
        return FixedpointPaillierVector.zeros(size)

    @staticmethod
    def i_add(pk: PK, a: EV, b: EV, sa=0, sb=0, size: Optional[int] = None) -> None:
        """
        inplace add, a[sa:sa+size] += b[sb:sb+size], if size is None, then size = min(a.size - sa, b.size - sb)
        Args:
            pk: the public key
            a: the vector to add to
            b: the vector to add
            sa: the start index of a
            sb: the start index of b
            size: the size to add
        """
        if a is b:
            a.iadd_vec_self(sa, sb, size, pk.pk)
        else:
            a.iadd_vec(b, sa, sb, size, pk.pk)

    @staticmethod
    def slice(a: EV, start: int, size: int) -> EV:
        """
        slice a[start:start+size]
        Args:
            a: the vector to slice
            start: the start index
            size: the size to slice

        Returns:
            the sliced vector
        """
        return a.slice(start, size)

    @staticmethod
    def i_shuffle(pk: PK, a: EV, indices: torch.LongTensor) -> None:
        """
        inplace shuffle, a = a[indices]
        Args:
            pk: public key, not used
            a: the vector to shuffle
            indices: the indices to shuffle
        """
        a.i_shuffle(indices)

    @staticmethod
    def intervals_slice(a: EV, intervals: List[Tuple[int, int]]) -> EV:
        """
        slice in the given intervals

        for example:
            intervals=[(0, 4), (6, 12)], a = [a0, a1, a2, a3, a4, a5, a6, a7,...]
            then the result is [a0, a1, a2, a3, a6, a7, a8, a9, a10, a11]
        """
        return a.intervals_slice(intervals)

    @staticmethod
    def slice_indexes(a: EV, indexes: List[int]) -> EV:
        """
        slice in the given indexes
        Args:
            a:
            indexes:

        Returns:

        """
        return a.slice_indexes(indexes)

    @staticmethod
    def cat(list: List[EV]) -> EV:
        """
        concatenate the list of vectors
        Args:
            list: the list of vectors

        Returns: the concatenated vector
        """
        return list[0].cat(list[1:])

    @staticmethod
    def intervals_sum_with_step(pk: PK, a: EV, intervals: List[Tuple[int, int]], step: int):
        """
        sum in the given intervals, with step size

        for example:
            if step=2, intervals=[(0, 4), (6, 12)], a = [a0, a1, a2, a3, a4, a5, a6, a7,...]
            then the result is [a0+a2, a1+a3, a6+a8+a10, a7+a9+a11]
        """
        return a.intervals_sum_with_step(pk.pk, intervals, step)

    @staticmethod
    def chunking_cumsum_with_step(pk: PK, a: EV, chunk_sizes: List[int], step: int):
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
