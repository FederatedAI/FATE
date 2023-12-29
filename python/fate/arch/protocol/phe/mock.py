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

from .type import TensorEvaluator

V = torch.Tensor


class EV:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return f"<EV {self.data}>"

    def __repr__(self):
        return str(self)

    def tolist(self):
        return [EV(x.clone().detach()) for x in self.data]


class FV:
    def __init__(self, data):
        self.data = data


class SK:
    def __init__(self):
        ...

    def decrypt_to_encoded(self, vec: EV) -> FV:
        return FV(vec.data)


class PK:
    def __init__(self):
        ...

    def encrypt_encoded(self, vec: FV, obfuscate: bool) -> EV:
        return EV(vec.data)

    def encrypt_encoded_scalar(self, val, obfuscate) -> EV:
        return EV(val)


class Coder:
    def __init__(self):
        ...

    def pack_floats(self, float_tensor: V, offset_bit: int, pack_num: int, precision: int) -> FV:
        return float_tensor

    def unpack_floats(self, packed: FV, offset_bit: int, pack_num: int, precision: int, total_num: int) -> V:
        return torch.tensor(self.coder.unpack_floats(packed, offset_bit, pack_num, precision, total_num))

    def pack_vec(self, vec: torch.LongTensor, num_shift_bit, num_elem_each_pack) -> FV:
        return self.coder.pack_u64_vec(vec.detach().tolist(), num_shift_bit, num_elem_each_pack)

    def unpack_vec(self, vec: FV, num_shift_bit, num_elem_each_pack, total_num) -> torch.LongTensor:
        return torch.LongTensor(self.coder.unpack_u64_vec(vec, num_shift_bit, num_elem_each_pack, total_num))

    def encode_tensor(self, tensor: V, dtype: torch.dtype = None) -> FV:
        if dtype is None:
            dtype = tensor.dtype
        return self.encode_vec(tensor.flatten(), dtype=dtype)

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
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
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
        raise NotImplementedError(f"{dtype} not supported, val={val}, type={type(val)}")

    def encode_f64(self, val: float) -> FV:
        return torch.tensor(val, dtype=torch.float64)

    def decode_f64(self, val):
        return float(val.item())

    def encode_i64(self, val: int):
        return torch.tensor(val, dtype=torch.int64)

    def decode_i64(self, val):
        return int(val.item())

    def encode_f32(self, val: float):
        return torch.tensor(val, dtype=torch.float32)

    def decode_f32(self, val):
        return float(val.item())

    def encode_i32(self, val: int):
        return torch.tensor(val, dtype=torch.int32)

    def decode_i32(self, val):
        return int(val.item())

    def encode_f64_vec(self, vec: torch.Tensor):
        return FV(vec.detach().flatten())

    def decode_f64_vec(self, vec):
        return vec.data.type(torch.float64)

    def encode_i64_vec(self, vec: torch.Tensor):
        return FV(vec.detach().flatten())

    def decode_i64_vec(self, vec):
        return vec.data.type(torch.int64)

    def encode_f32_vec(self, vec: torch.Tensor):
        return FV(vec.detach().flatten())

    def decode_f32_vec(self, vec):
        return vec.data.type(torch.float32)

    def encode_i32_vec(self, vec: torch.Tensor):
        return FV(vec.detach().flatten())

    def decode_i32_vec(self, vec):
        return vec.data.type(torch.int32)


def keygen(key_size):
    return SK(), PK(), Coder()


class evaluator(TensorEvaluator[EV, V, PK, Coder]):
    @staticmethod
    def add(a: EV, b: EV, pk: PK):
        return EV(torch.add(a.data, b.data))

    @staticmethod
    def add_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        return EV(torch.add(a.data, pk.encrypt_encoded(coder.encode_tensor(b), obfuscate=False).data))

    @staticmethod
    def add_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        return EV(torch.add(a.data, pk.encrypt_encoded_scalar(coder.encode(b), obfuscate=False).data))

    @staticmethod
    def sub(a: EV, b: EV, pk: PK):
        return EV(torch.sub(a.data, b.data))

    @staticmethod
    def sub_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        data = torch.sub(a.data, pk.encrypt_encoded(coder.encode_tensor(b), obfuscate=False).data)
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def sub_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        data = torch.sub(a.data, pk.encrypt_encoded_scalar(coder.encode(b), obfuscate=False).data)
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def rsub(a: EV, b: EV, pk: PK):
        return EV(torch.rsub(a.data, b.data))

    @staticmethod
    def rsub_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        data = torch.rsub(a.data, pk.encrypt_encoded(coder.encode_tensor(b), obfuscate=False).data)
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def rsub_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        data = torch.rsub(a.data, pk.encrypt_encoded_scalar(coder.encode(b), obfuscate=False).data)
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def mul_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None):
        data = torch.mul(a.data, coder.encode_tensor(b).data)
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def mul_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype):
        data = torch.mul(a.data, coder.encode(b).data)
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def matmul(a: EV, b: V, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        left = a.data.reshape(a_shape)
        right = b.data.reshape(b_shape)
        target_type = torch.promote_types(a.data.dtype, b.data.dtype)
        if left.dtype != target_type:
            left = left.to(dtype=target_type)
        if right.dtype != target_type:
            right = right.to(dtype=target_type)
        data = torch.matmul(left, right).flatten()
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def rmatmul(a: EV, b: V, a_shape, b_shape, pk: PK, coder: Coder, output_dtype):
        right = a.data.reshape(a_shape)
        left = b.data.reshape(b_shape)
        target_type = torch.promote_types(a.data.dtype, b.data.dtype)
        if left.dtype != target_type:
            left = left.to(dtype=target_type)
        if right.dtype != target_type:
            right = right.to(dtype=target_type)
        data = torch.matmul(left, right).flatten()
        if output_dtype is not None:
            data = data.to(dtype=output_dtype)
        return EV(data)

    @staticmethod
    def zeros(size, dtype) -> EV:
        return EV(torch.zeros(size, dtype=dtype))

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
        if size is None:
            size = min(a.data.numel() - sa, b.data.numel() - sb)
        a.data[sa : sa + size] += b.data[sb : sb + size]

    @staticmethod
    def i_sub(pk: PK, a: EV, b: EV, sa=0, sb=0, size: Optional[int] = None) -> None:
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
        if size is None:
            size = min(a.data.numel() - sa, b.data.numel() - sb)
        a.data[sa : sa + size] -= b.data[sb : sb + size]

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
        return EV(a.data[start : start + size])

    @staticmethod
    def i_shuffle(pk: PK, a: EV, indices: torch.LongTensor) -> None:
        """
        inplace shuffle, a = a[indices]
        Args:
            pk: public key, not used
            a: the vector to shuffle
            indices: the indices to shuffle
        """
        shuffled = a.data[indices]
        a.data.copy_(shuffled)

    @staticmethod
    def shuffle(pk: PK, a: EV, indices: torch.LongTensor) -> EV:
        """
        inplace shuffle, a = a[indices]
        Args:
            pk: public key, not used
            a: the vector to shuffle
            indices: the indices to shuffle
        """
        shuffled = a.data[indices]
        return EV(shuffled)

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
        if stride == 1:
            index = torch.LongTensor(positions)
            value = b.data.view(-1, 1).expand(-1, index.shape[1]).flatten()
            index = index.flatten()
            data = a.data
        else:
            index = torch.LongTensor(positions)
            data = a.data.view(-1, stride)
            value = b.data.view(-1, stride).unsqueeze(1).expand(-1, index.shape[1], stride).reshape(-1, stride)
            index = index.flatten().unsqueeze(1).expand(-1, stride)
        try:
            data.scatter_add_(0, index, value)
        except Exception as e:
            raise ValueError(f"data: {data.dtype}, value: {value.dtype}") from e

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
        if stride == 1:
            b = b.data[masks]
            index = torch.LongTensor(positions)
            value = b.data.view(-1, 1).expand(-1, index.shape[1]).flatten()
            index = index.flatten()
            data = a.data
        else:
            index = torch.LongTensor(positions)
            data = a.data.view(-1, stride)
            value = b.data.view(-1, stride)[masks]
            value = value.unsqueeze(1).expand(-1, index.shape[1], stride).reshape(-1, stride)
            index = index.flatten().unsqueeze(1).expand(-1, stride)
        data.scatter_add_(0, index, value)

    @staticmethod
    def intervals_slice(a: EV, intervals: List[Tuple[int, int]]) -> EV:
        """
        slice in the given intervals

        for example:
            intervals=[(0, 4), (6, 12)], a = [a0, a1, a2, a3, a4, a5, a6, a7,...]
            then the result is [a0, a1, a2, a3, a6, a7, a8, a9, a10, a11]
        """
        slices = []
        for start, end in intervals:
            slices.append(a.data[start:end])
        return EV(torch.cat(slices))

    @staticmethod
    def cat(list: List[EV]) -> EV:
        """
        concatenate the list of vectors
        Args:
            list: the list of vectors

        Returns: the concatenated vector
        """

        if list[0].data.dim() == 0:
            return EV(torch.tensor([x.data for x in list]))
        return EV(torch.cat([x.data for x in list]))

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
        data_view = a.data.view(-1, step)
        start = 0
        for num in chunk_sizes:
            num = num // step
            data_view[start : start + num, :] = data_view[start : start + num, :].cumsum(dim=0)
            start += num

    @staticmethod
    def pack_squeeze(a: EV, pack_num: int, shift_bit: int, pk: PK) -> EV:
        return a.pack_squeeze(pack_num, shift_bit, pk.pk)
