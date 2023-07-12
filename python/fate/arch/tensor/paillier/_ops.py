import torch
from fate.arch.tensor import _custom_ops

from ._tensor import PaillierTensor, implements


@implements(_custom_ops.decrypt_f)
def decrypt(input, decryptor):
    return decryptor.decrypt(input)


@implements(torch.add)
def add(input: PaillierTensor, other):
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return add(other, input)

    if isinstance(other, PaillierTensor):
        return PaillierTensor(input._data.add_cipherblock(other._data), torch.promote_types(input.dtype, other.dtype))
    if isinstance(other, torch.Tensor):
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if other.dtype == torch.float64:
            return PaillierTensor(input._data.add_plaintext_f64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.float32:
            return PaillierTensor(input._data.add_plaintext_f32(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int64:
            return PaillierTensor(input._data.add_plaintext_i64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int32:
            return PaillierTensor(input._data.add_plaintext_i32(other.detach().numpy()), output_dtype)
        return NotImplemented
    if isinstance(other, (float, int)):
        if input.dtype == torch.float64:
            return PaillierTensor(input._data.add_plaintext_scalar_f64(float(other)), torch.float64)
        elif input.dtype == torch.float32:
            return PaillierTensor(input._data.add_plaintext_scalar_f32(float(other)), torch.float32)
        elif input.dtype == torch.int64 and isinstance(other, int):
            return PaillierTensor(input._data.add_plaintext_scalar_i64(other), torch.int64)
        elif input.dtype == torch.int32 and isinstance(other, int):
            return PaillierTensor(input._data.add_plaintext_scalar_i32(other), torch.int32)
        else:
            raise NotImplementedError(f"operation on encrypt {input.dtype} with `{type(other)}` maybe problematic")
    return NotImplemented


@implements(torch.rsub)
def rsub(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return sub(other, input)

    if isinstance(other, PaillierTensor):
        return PaillierTensor(input._data.sub_cipherblock(other._data), torch.promote_types(input.dtype, other.dtype))

    if isinstance(other, torch.Tensor):
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if other.dtype == torch.float64:
            return PaillierTensor(input._data.rsub_plaintext_f64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.float32:
            return PaillierTensor(input._data.rsub_plaintext_f32(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int64:
            return PaillierTensor(input._data.rsub_plaintext_i64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int32:
            return PaillierTensor(input._data.rsub_plaintext_i32(other.detach().numpy()), output_dtype)
        return NotImplemented
    if isinstance(other, (float, int)):
        if input.dtype == torch.float64:
            return PaillierTensor(input._data.rsub_plaintext_scalar_f64(float(other)), torch.float64)
        elif input.dtype == torch.float32:
            return PaillierTensor(input._data.rsub_plaintext_scalar_f32(float(other)), torch.float32)
        elif input.dtype == torch.int64 and isinstance(other, int):
            return PaillierTensor(input._data.rsub_plaintext_scalar_i64(other), torch.int64)
        elif input.dtype == torch.int32 and isinstance(other, int):
            return PaillierTensor(input._data.rsub_plaintext_scalar_i32(other), torch.int32)
        else:
            raise NotImplementedError(f"operation on encrypt {input.dtype} with `{type(other)}` maybe problematic")
    return NotImplemented


@implements(torch.sub)
def sub(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return rsub(other, input)

    if isinstance(other, PaillierTensor):
        return PaillierTensor(input._data.sub_cipherblock(other._data), torch.promote_types(input.dtype, other.dtype))

    if isinstance(other, torch.Tensor):
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if other.dtype == torch.float64:
            return PaillierTensor(input._data.sub_plaintext_f64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.float32:
            return PaillierTensor(input._data.sub_plaintext_f32(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int64:
            return PaillierTensor(input._data.sub_plaintext_i64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int32:
            return PaillierTensor(input._data.sub_plaintext_i32(other.detach().numpy()), output_dtype)
        return NotImplemented
    if isinstance(other, (float, int)):
        if input.dtype == torch.float64:
            return PaillierTensor(input._data.sub_plaintext_scalar_f64(float(other)), torch.float64)
        elif input.dtype == torch.float32:
            return PaillierTensor(input._data.sub_plaintext_scalar_f32(float(other)), torch.float32)
        elif input.dtype == torch.int64 and isinstance(other, int):
            return PaillierTensor(input._data.sub_plaintext_scalar_i64(other), torch.int64)
        elif input.dtype == torch.int32 and isinstance(other, int):
            return PaillierTensor(input._data.sub_plaintext_scalar_i32(other), torch.int32)
        else:
            raise NotImplementedError(f"operation on encrypt {input.dtype} with `{type(other)}` maybe problematic")
    return NotImplemented


@implements(torch.mul)
def mul(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return mul(other, input)

    if isinstance(other, PaillierTensor):
        raise ValueError("can't mul `PaillierTensor` with `PaillierTensor`")

    if isinstance(other, torch.Tensor):
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if other.dtype == torch.float64:
            return PaillierTensor(input._data.mul_plaintext_f64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.float32:
            return PaillierTensor(input._data.mul_plaintext_f32(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int64:
            return PaillierTensor(input._data.mul_plaintext_i64(other.detach().numpy()), output_dtype)
        if other.dtype == torch.int32:
            return PaillierTensor(input._data.mul_plaintext_i32(other.detach().numpy()), output_dtype)
        return NotImplemented
    if isinstance(other, (float, int)):
        if input.dtype == torch.float64:
            return PaillierTensor(input._data.mul_plaintext_scalar_f64(float(other)), torch.float64)
        elif input.dtype == torch.float32:
            return PaillierTensor(input._data.mul_plaintext_scalar_f32(float(other)), torch.float32)
        elif input.dtype == torch.int64 and isinstance(other, int):
            return PaillierTensor(input._data.mul_plaintext_scalar_i64(other), torch.int64)
        elif input.dtype == torch.int32 and isinstance(other, int):
            return PaillierTensor(input._data.mul_plaintext_scalar_i32(other), torch.int32)
        else:
            raise NotImplementedError(f"operation on encrypt {input.dtype} with `{type(other)}` maybe problematic")
    return NotImplemented


@implements(_custom_ops.rmatmul_f)
def rmatmul_f(input, other):
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return matmul(other, input)

    if isinstance(other, torch.Tensor):
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if len(other.shape) == 1:
            if other.dtype == torch.float64:
                return PaillierTensor(input._data.rmatmul_plaintext_ix1_f64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.float32:
                return PaillierTensor(input._data.rmatmul_plaintext_ix1_f32(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int64:
                return PaillierTensor(input._data.rmatmul_plaintext_ix1_i64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int32:
                return PaillierTensor(input._data.rmatmul_plaintext_ix1_i32(other.detach().numpy()), output_dtype)
            return NotImplemented
        elif len(other.shape) == 2:
            if other.dtype == torch.float64:
                return PaillierTensor(input._data.rmatmul_plaintext_ix2_f64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.float32:
                return PaillierTensor(input._data.rmatmul_plaintext_ix2_f32(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int64:
                return PaillierTensor(input._data.rmatmul_plaintext_ix2_i64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int32:
                return PaillierTensor(input._data.rmatmul_plaintext_ix2_i32(other.detach().numpy()), output_dtype)
            return NotImplemented
        else:
            raise ValueError(f"can't matmul `PaillierTensor` with `torch.Tensor` with dim `{len(other.shape)}`")
    return NotImplemented


@implements(torch.matmul)
def matmul(input, other):
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return rmatmul_f(other, input)

    if isinstance(other, PaillierTensor):
        raise ValueError("can't matmul `PaillierTensor` with `PaillierTensor`")

    if isinstance(other, torch.Tensor):
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if len(other.shape) == 1:
            if other.dtype == torch.float64:
                return PaillierTensor(input._data.matmul_plaintext_ix1_f64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.float32:
                return PaillierTensor(input._data.matmul_plaintext_ix1_f32(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int64:
                return PaillierTensor(input._data.matmul_plaintext_ix1_i64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int32:
                return PaillierTensor(input._data.matmul_plaintext_ix1_i32(other.detach().numpy()), output_dtype)
            return NotImplemented
        elif len(other.shape) == 2:
            if other.dtype == torch.float64:
                return PaillierTensor(input._data.matmul_plaintext_ix2_f64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.float32:
                return PaillierTensor(input._data.matmul_plaintext_ix2_f32(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int64:
                return PaillierTensor(input._data.matmul_plaintext_ix2_i64(other.detach().numpy()), output_dtype)
            if other.dtype == torch.int32:
                return PaillierTensor(input._data.matmul_plaintext_ix2_i32(other.detach().numpy()), output_dtype)
            return NotImplemented
        else:
            raise ValueError(f"can't matmul `PaillierTensor` with `torch.Tensor` with dim `{len(other.shape)}`")
    return NotImplemented


@implements(_custom_ops.to_local_f)
def to_local_f(input):
    return input
