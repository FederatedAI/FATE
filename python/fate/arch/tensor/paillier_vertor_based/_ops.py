import torch
from fate.arch.protocol.paillier import ops
from fate.arch.tensor import _custom_ops

from ._tensor import PaillierTensor, implements


@implements(_custom_ops.decrypt_f)
def decrypt(input, decryptor):
    return decryptor.decrypt(input)


@implements(torch.add)
def add(input: PaillierTensor, other):
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return add(other, input)

    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PaillierTensor):
        assert shape == other.shape, f"shape mismatch {shape} != {other.shape}"
        output_dtype = torch.promote_types(dtype, other.dtype)
        data = ops.add(input.data, other.data, pk)
        return input.with_template(data, dtype=output_dtype)

    elif isinstance(other, torch.Tensor):
        # TODO: support broadcast
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.add_vec(input.data, other.flatten().detach(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.add_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = ops.add_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)
    else:
        return NotImplemented


@implements(torch.rsub)
def rsub(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return sub(other, input)

    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PaillierTensor):
        assert shape == other.shape, f"shape mismatch {shape} != {other.shape}"
        output_dtype = torch.promote_types(dtype, other.dtype)
        data = ops.rsub(input.data, other.data, pk)
        return input.with_template(data, dtype=output_dtype)

    elif isinstance(other, torch.Tensor):
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.rsub_vec(input.data, other.flatten().detach(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.rsub_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = ops.rsub_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)

    else:
        return NotImplemented


@implements(torch.sub)
def sub(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return rsub(other, input)

    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PaillierTensor):
        assert shape == other.shape, f"shape mismatch {shape} != {other.shape}"
        output_dtype = torch.promote_types(dtype, other.dtype)
        data = ops.sub(input.data, other.data, pk)
        return input.with_template(data, dtype=output_dtype)

    elif isinstance(other, torch.Tensor):
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.sub_vec(input.data, other.flatten().detach(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.sub_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = ops.sub_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)

    else:
        return NotImplemented


@implements(torch.mul)
def mul(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return mul(other, input)

    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PaillierTensor):
        raise NotImplementedError(
            f"mul {input} with {other} not supported, paillier is not multiplicative homomorphic"
        )

    elif isinstance(other, torch.Tensor):
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.mul_vec(input.data, other.flatten().detach(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = ops.mul_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = ops.mul_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)

    else:
        return NotImplemented


@implements(_custom_ops.rmatmul_f)
def rmatmul_f(input, other):
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return matmul(other, input)

    if input.ndim > 2 or input.ndim < 1:
        raise ValueError(f"can't rmatmul `PaillierTensor` with `torch.Tensor` with dim `{input.ndim}`")

    if isinstance(other, PaillierTensor):
        raise NotImplementedError(
            f"rmatmul {input} with {other} not supported, paillier is not multiplicative homomorphic"
        )

    if not isinstance(other, torch.Tensor):
        return NotImplemented

    pk = input.pk
    coder = input.coder
    shape = input.shape
    other_shape = other.shape
    output_dtype = torch.promote_types(input.dtype, other.dtype)
    output_shape = torch.matmul(torch.rand(*other_shape, device="meta"), torch.rand(*shape, device="meta")).shape
    data = ops.rmatmul(input.data, other.flatten().detach(), shape, other_shape, pk, coder, output_dtype)
    return PaillierTensor(pk, coder, output_shape, data, output_dtype)


@implements(torch.matmul)
def matmul(input, other):
    if not isinstance(input, PaillierTensor) and isinstance(other, PaillierTensor):
        return rmatmul_f(other, input)

    if input.ndim > 2 or input.ndim < 1:
        raise ValueError(f"can't matmul `PaillierTensor` with `torch.Tensor` with dim `{input.ndim}`")

    if isinstance(other, PaillierTensor):
        raise ValueError("can't matmul `PaillierTensor` with `PaillierTensor`")

    if not isinstance(other, torch.Tensor):
        return NotImplemented

    pk = input.pk
    coder = input.coder
    shape = input.shape
    other_shape = other.shape
    output_dtype = torch.promote_types(input.dtype, other.dtype)
    output_shape = torch.matmul(torch.rand(*shape, device="meta"), torch.rand(*other_shape, device="meta")).shape
    data = ops.matmul(input.data, other.flatten().detach(), shape, other_shape, pk, coder, output_dtype)
    return PaillierTensor(pk, coder, output_shape, data, output_dtype)


@implements(_custom_ops.to_local_f)
def to_local_f(input):
    return input
