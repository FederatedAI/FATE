#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List, Optional, Tuple

import torch
from fate_utils.ou import CiphertextVector, PlaintextVector
from fate_utils.ou import Coder as _Coder
from fate_utils.ou import Evaluator as _Evaluator
from fate_utils.ou import PK as _PK
from fate_utils.ou import SK as _SK
from fate_utils.ou import keygen as _keygen

from .type import TensorEvaluator

V = torch.Tensor
EV = CiphertextVector
FV = PlaintextVector


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

    def encrypt_encoded_scalar(self, val, obfuscate) -> EV:
        return self.pk.encrypt_encoded_scalar(val, obfuscate)


class Coder:
    def __init__(self, coder: _Coder):
        self.coder = coder

    def pack_floats(self, float_tensor: V, offset_bit: int, pack_num: int, precision: int) -> FV:
        return self.coder.pack_floats(float_tensor.detach().tolist(), offset_bit, pack_num, precision)

    def unpack_floats(self, packed: FV, offset_bit: int, pack_num: int, precision: int, total_num: int) -> V:
        return torch.tensor(self.coder.unpack_floats(packed, offset_bit, pack_num, precision, total_num))

    def pack_vec(self, vec: torch.LongTensor, num_shift_bit, num_elem_each_pack) -> FV:
        return self.coder.pack_u64_vec(vec.detach().tolist(), num_shift_bit, num_elem_each_pack)

    def unpack_vec(self, vec: FV, num_shift_bit, num_elem_each_pack, total_num) -> torch.LongTensor:
        return torch.LongTensor(self.coder.unpack_u64_vec(vec, num_shift_bit, num_elem_each_pack, total_num))

    def encode_tensor(self, tensor: V, dtype: torch.dtype = None) -> FV:
        return self.encode_vec(tensor.flatten(), dtype=tensor.dtype)

    def decode_tensor(self, tensor: FV, dtype: torch.dtype, shape: torch.Size = None, device=None) -> V:
        data = self.decode_vec(tensor, dtype)
        if shape is not None:
            data = data.reshape(shape)
        if device is not None:
            data = data.to(device.to_torch_device())
        return data

    def encode_vec(self, vec: V, dtype: torch.dtype = None) -> FV:
        if dtype is None:
            dtype = vec.dtype
        else:
            if dtype != vec.dtype:
                vec = vec.to(dtype=dtype)
        # if dtype == torch.float64:
        #     return self.encode_f64_vec(vec)
        # if dtype == torch.float32:
        #     return self.encode_f32_vec(vec)
        if dtype == torch.int64:
            return self.encode_i64_vec(vec)
        if dtype == torch.int32:
            return self.encode_i32_vec(vec)
        raise NotImplementedError(f"{vec.dtype} not supported")

    def decode_vec(self, vec: FV, dtype: torch.dtype) -> V:
        # if dtype == torch.float64:
        #     return self.decode_f64_vec(vec)
        # if dtype == torch.float32:
        #     return self.decode_f32_vec(vec)
        if dtype == torch.int64:
            return self.decode_i64_vec(vec)
        if dtype == torch.int32:
            return self.decode_i32_vec(vec)
        raise NotImplementedError(f"{dtype} not supported")

    def encode(self, val, dtype=None) -> FV:
        if isinstance(val, torch.Tensor):
            assert val.ndim == 0, "only scalar supported"
            dtype = val.dtype
            val = val.item()
        # if dtype == torch.float64:
        #     return self.encode_f64(val)
        # if dtype == torch.float32:
        #     return self.encode_f32(val)
        if dtype == torch.int64:
            return self.encode_i64(val)
        if dtype == torch.int32:
            return self.encode_i32(val)
        raise NotImplementedError(f"{dtype} not supported")

    # def encode_f64(self, val: float):
    #     return self.coder.encode_f64(val)
    #
    # def decode_f64(self, val):
    #     return self.coder.decode_f64(val)

    def encode_i64(self, val: int):
        return self.coder.encode_u64(val)

    def decode_i64(self, val):
        return self.coder.decode_u64(val)

    # def encode_f32(self, val: float):
    #     return self.coder.encode_f32(val)
    #
    # def decode_f32(self, val):
    #     return self.coder.decode_f32(val)

    def encode_i32(self, val: int):
        return self.coder.encode_u32(val)

    def decode_i32(self, val):
        return self.coder.decode_u32(val)

    # def encode_f64_vec(self, vec: torch.Tensor):
    #     vec = vec.detach().flatten()
    #     return self.coder.encode_f64_vec(vec.detach().numpy())
    #
    # def decode_f64_vec(self, vec):
    #     return torch.tensor(self.coder.decode_f64_vec(vec))

    def encode_i64_vec(self, vec: torch.Tensor):
        vec = vec.detach().flatten()
        return self.coder.encode_u64_vec(vec.detach().numpy().astype("uint64"))

    def decode_i64_vec(self, vec):
        return torch.tensor(self.coder.decode_u64_vec(vec))

    # def encode_f32_vec(self, vec: torch.Tensor):
    #     vec = vec.detach().flatten()
    #     return self.coder.encode_f32_vec(vec.detach().numpy())
    #
    # def decode_f32_vec(self, vec):
    #     return torch.tensor(self.coder.decode_f32_vec(vec))

    def encode_i32_vec(self, vec: torch.Tensor):
        vec = vec.detach().flatten()
        return self.coder.encode_u32_vec(vec.detach().numpy().astype("uint32"))

    def decode_i32_vec(self, vec):
        return torch.tensor(self.coder.decode_u32_vec(vec))


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
    def zeros(size, dtype) -> EV:
        return CiphertextVector.zeros(size)

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
    def i_sub(pk: PK, a: EV, b: EV, sa=0, sb=0, size: Optional[int] = None) -> None:
        """
        inplace sub, a[sa:sa+size] += b[sb:sb+size], if size is None, then size = min(a.size - sa, b.size - sb)
        Args:
            pk: the public key
            a: the vector to add to
            b: the vector to add
            sa: the start index of a
            sb: the start index of b
            size: the size to add
        """
        if a is b:
            a.isub_vec_self(sa, sb, size, pk.pk)
        else:
            a.isub_vec(b, sa, sb, size, pk.pk)

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
    def shuffle(pk: PK, a: EV, indices: torch.LongTensor) -> EV:
        """
        shuffle, out = a[indices]
        Args:
            pk: public key, not used
            a: the vector to shuffle
            indices: the indices to shuffle
        """
        return a.shuffle(indices)

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
        a.iupdate(b, positions, stride, pk.pk)

    @staticmethod
    def i_update_with_masks(pk: PK, a: EV, b: EV, positions, masks, stride: int) -> None:
        """
        inplace update, a[positions] += b[::stride]
        Args:
            pk: public key, not used
            a: the vector to update
            b: the vector to update with
            positions: the positions to update
            stride: the stride to update
        """
        a.iupdate_with_masks(b, positions, masks, stride, pk.pk)

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
    def cat(list: List[EV]) -> EV:
        """
        concatenate the list of vectors
        Args:
            list: the list of vectors

        Returns: the concatenated vector
        """
        return _Evaluator.cat(list)

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

    @staticmethod
    def pack_squeeze(a: EV, pack_num: int, shift_bit: int, pk: PK) -> EV:
        return a.pack_squeeze(pack_num, shift_bit, pk.pk)
