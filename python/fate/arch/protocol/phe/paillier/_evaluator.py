from typing import Optional, List, Tuple

from fate_utils.paillier import ArithmeticEvaluator, BaseEvaluator, HistogramEvaluator

from fate.arch.protocol.phe.type import TensorEvaluator
from ._cipher import E, F, V, EV, FV, PK, SK
from ._coder import Coder


class Evaluator(TensorEvaluator[EV, V, PK, Coder]):

    @staticmethod
    def encrypt(pk: PK, vec: FV, obfuscate: bool) -> EV:
        return BaseEvaluator.encrypt(vec, obfuscate, pk.pk)

    @staticmethod
    def encrypt_scalar(pk: PK, val, obfuscate: bool) -> E:
        return BaseEvaluator.encrypt_scalar(val, obfuscate, pk.pk)

    @staticmethod
    def decrypt(sk: SK, vec: EV) -> FV:
        return BaseEvaluator.decrypt(vec, sk.sk)

    @staticmethod
    def decrypt_scalar(sk: SK, vec: E) -> F:
        return BaseEvaluator.decrypt_scalar(vec, sk.sk)

    @staticmethod
    def add(a: EV, b: EV, pk: PK):
        return ArithmeticEvaluator.add(a, b, pk.pk)

    @staticmethod
    def add_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded(encoded, obfuscate=False)
        return ArithmeticEvaluator.add(a, encrypted, pk.pk)

    @staticmethod
    def add_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return ArithmeticEvaluator.add_scalar(a, encrypted, pk.pk)

    @staticmethod
    def sub(a: EV, b: EV, pk: PK):
        return ArithmeticEvaluator.sub(a, b, pk.pk)

    @staticmethod
    def sub_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded(encoded, obfuscate=False)
        return ArithmeticEvaluator.sub(a, encrypted, pk.pk)

    @staticmethod
    def sub_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return ArithmeticEvaluator.sub_scalar(a, encrypted, pk.pk)

    @staticmethod
    def rsub(a: EV, b: EV, pk: PK):
        return ArithmeticEvaluator.rsub(a, b, pk.pk)

    @staticmethod
    def rsub_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded(encoded, obfuscate=False)
        return ArithmeticEvaluator.rsub(a, encrypted, pk.pk)

    @staticmethod
    def rsub_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return ArithmeticEvaluator.rsub_scalar(a, encrypted, pk.pk)

    @staticmethod
    def mul_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        if output_dtype is None:
            output_dtype = b.dtype
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        return ArithmeticEvaluator.mul(a, encoded, pk.pk)

    @staticmethod
    def mul_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode(b, dtype=output_dtype)
        return ArithmeticEvaluator.mul_scalar(a, encoded, pk.pk)

    @staticmethod
    def sum(a: EV, shape: List[int], dim: Optional[int], pk: PK) -> EV:
        return ArithmeticEvaluator.sum(a, shape, dim, pk.pk)

    @staticmethod
    def matmul(a: EV, b: V, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        # TODO: move this to python side so other protocols can use it without matmul support?
        return ArithmeticEvaluator.matmul(a, encoded, a_shape, b_shape, pk.pk)

    @staticmethod
    def rmatmul(a: EV, b: V, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        encoded = coder.encode_tensor(b, dtype=output_dtype)
        return ArithmeticEvaluator.rmatmul(a, encoded, a_shape, b_shape, pk.pk)

    @staticmethod
    def zeros(size) -> EV:
        return ArithmeticEvaluator.zeros(size)

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
            HistogramEvaluator.iadd_vec_self(a, sa, sb, size, pk.pk)
        else:
            HistogramEvaluator.iadd_vec(a, b, sa, sb, size, pk.pk)

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
        return HistogramEvaluator.slice(a, start, size)

    @staticmethod
    def i_shuffle(pk: PK, a: EV, indices: List[int]) -> None:
        """
        inplace shuffle, a = a[indices]
        Args:
            pk: public key, not used
            a: the vector to shuffle
            indices: the indices to shuffle
        """
        HistogramEvaluator.i_shuffle(a, indices)

    @staticmethod
    def i_update(pk: PK, a: EV, b: EV, positions, stride: int) -> None:
        """
        inplace update, a[positions] += b[::stride]
        Args:
            pk: public key, not used
            a: the vector to update
            b: the vector to update with
            positions: the positions to update
            stride: the stride to update
        """
        HistogramEvaluator.iupdate(a, b, positions, stride, pk.pk)

    @staticmethod
    def intervals_slice(a: EV, intervals: List[Tuple[int, int]]) -> EV:
        """
        slice in the given intervals

        for example:
            intervals=[(0, 4), (6, 12)], a = [a0, a1, a2, a3, a4, a5, a6, a7,...]
            then the result is [a0, a1, a2, a3, a6, a7, a8, a9, a10, a11]
        """
        return HistogramEvaluator.intervals_slice(a, intervals)

    @staticmethod
    def slice_indexes(a: EV, indexes: List[int]) -> EV:
        """
        slice in the given indexes
        Args:
            a:
            indexes:

        Returns:

        """
        return HistogramEvaluator.slice_indexes(a, indexes)

    @staticmethod
    def cat(list_vec: List[EV]) -> EV:
        """
        concatenate the list of vectors
        Args:
            list_vec: the list of vectors

        Returns: the concatenated vector
        """
        return BaseEvaluator.cat(list_vec)

    @staticmethod
    def intervals_sum_with_step(pk: PK, a: EV, intervals: List[Tuple[int, int]], step: int):
        """
        sum in the given intervals, with step size

        for example:
            if step=2, intervals=[(0, 4), (6, 12)], a = [a0, a1, a2, a3, a4, a5, a6, a7,...]
            then the result is [a0+a2, a1+a3, a6+a8+a10, a7+a9+a11]
        """
        return HistogramEvaluator.intervals_sum_with_step(a, intervals, step, pk.pk)

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
        return HistogramEvaluator.chunking_cumsum_with_step(a, chunk_sizes, step, pk.pk)

    @staticmethod
    def pack_squeeze(a: EV, pack_num: int, shift_bit: int, pk: PK) -> EV:
        return HistogramEvaluator.pack_squeeze(a, pack_num, shift_bit, pk.pk)
