import torch
from fate_utils.paillier import Coder as _Coder
from fate_utils.paillier import Packer

from ._cipher import V, FV, F, PK


class Coder:
    def __init__(self, coder: _Coder):
        self.coder = coder

    @classmethod
    def from_pk(cls, pk: PK):
        return cls(_Coder.from_pk(pk.pk))

    @staticmethod
    def pack_floats(float_tensor: V, offset_bit: int, pack_num: int, precision: int) -> FV:
        return Packer.pack_floats(float_tensor.detach().tolist(), offset_bit, pack_num, precision)

    @staticmethod
    def unpack_floats(packed: FV, offset_bit: int, pack_num: int, precision: int, total_num: int) -> V:
        return torch.tensor(Packer.unpack_floats(packed, offset_bit, pack_num, precision, total_num))

    @staticmethod
    def pack_vec(vec: torch.LongTensor, num_shift_bit, num_elem_each_pack) -> FV:
        return Packer.pack_u64_vec(vec.detach().tolist(), num_shift_bit, num_elem_each_pack)

    @staticmethod
    def unpack_vec(vec: FV, num_shift_bit, num_elem_each_pack, total_num) -> torch.LongTensor:
        return torch.LongTensor(Packer.unpack_u64_vec(vec, num_shift_bit, num_elem_each_pack, total_num))

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

    def encode(self, val, dtype=None) -> F:
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
