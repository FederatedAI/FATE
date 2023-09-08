import torch
from fate.arch.tensor import _custom_ops

from ._tensor import PHETensor, implements, implements_encoded


@implements(_custom_ops.encrypt_f)
def encrypt(input, encryptor):
    return encryptor.encrypt_tensor(input)


@implements_encoded(_custom_ops.encrypt_encoded_f)
def encrypt_encoded(input, encryptor):
    return encryptor.encrypt_encoded(input)


@implements(_custom_ops.decrypt_encoded_f)
def decrypt_encoded(input, decryptor):
    return decryptor.decrypt_encoded(input)


@implements(_custom_ops.decrypt_f)
def decrypt(input, decryptor):
    return decryptor.decrypt_tensor(input)


@implements(_custom_ops.encode_f)
def encode(input, coder):
    return coder.encode(input)


@implements_encoded(_custom_ops.decode_f)
def decode(input, coder):
    return coder.decode(input)


@implements(torch.add)
def add(input: PHETensor, other):
    if not isinstance(input, PHETensor) and isinstance(other, PHETensor):
        return add(other, input)

    evaluator = input.evaluator
    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PHETensor):
        assert shape == other.shape, f"shape mismatch {shape} != {other.shape}"
        output_dtype = torch.promote_types(dtype, other.dtype)
        data = evaluator.add(input.data, other.data, pk)
        return input.with_template(data, dtype=output_dtype)

    elif isinstance(other, torch.Tensor):
        # TODO: support broadcast
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.add_plain(input.data, other, pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.add_plain_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = evaluator.add_plain_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)
    else:
        return NotImplemented


@implements(torch.rsub)
def rsub(input, other):
    if not isinstance(input, PHETensor) and isinstance(other, PHETensor):
        return sub(other, input)

    evaluator = input.evaluator
    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PHETensor):
        assert shape == other.shape, f"shape mismatch {shape} != {other.shape}"
        output_dtype = torch.promote_types(dtype, other.dtype)
        data = evaluator.rsub(input.data, other.data, pk)
        return input.with_template(data, dtype=output_dtype)

    elif isinstance(other, torch.Tensor):
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.rsub_plain(input.data, other, pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.rsub_plain_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = evaluator.rsub_plain_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)

    else:
        return NotImplemented


@implements(torch.sub)
def sub(input, other):
    if not isinstance(input, PHETensor) and isinstance(other, PHETensor):
        return rsub(other, input)

    evaluator = input.evaluator
    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PHETensor):
        assert shape == other.shape, f"shape mismatch {shape} != {other.shape}"
        output_dtype = torch.promote_types(dtype, other.dtype)
        data = evaluator.sub(input.data, other.data, pk)
        return input.with_template(data, dtype=output_dtype)

    elif isinstance(other, torch.Tensor):
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.sub_plain(input.data, other, pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.sub_plain_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = evaluator.sub_plain_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)

    else:
        return NotImplemented


@implements(torch.mul)
def mul(input, other):
    if not isinstance(input, PHETensor) and isinstance(other, PHETensor):
        return mul(other, input)

    evaluator = input.evaluator
    pk = input.pk
    coder = input.coder
    shape = input.shape
    dtype = input.dtype
    if isinstance(other, PHETensor):
        raise NotImplementedError(
            f"mul {input} with {other} not supported, paillier is not multiplicative homomorphic"
        )

    elif isinstance(other, torch.Tensor):
        if shape == other.shape:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.mul_plain(input.data, other, pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        elif other.ndim == 0:
            output_dtype = torch.promote_types(dtype, other.dtype)
            data = evaluator.mul_plain_scalar(input.data, other.detach().item(), pk, coder, output_dtype)
            return input.with_template(data, dtype=output_dtype)
        else:
            raise NotImplementedError(f"broadcast not supported")

    elif isinstance(other, (float, int)):
        output_dtype = torch.promote_types(dtype, torch.get_default_dtype())
        data = evaluator.mul_plain_scalar(input.data, other, pk, coder, output_dtype)
        return input.with_template(data, dtype=output_dtype)

    else:
        return NotImplemented


@implements(_custom_ops.rmatmul_f)
def rmatmul_f(input, other):
    if not isinstance(input, PHETensor) and isinstance(other, PHETensor):
        return matmul(other, input)

    if input.ndim > 2 or input.ndim < 1:
        raise ValueError(f"can't rmatmul `PHETensor` with `torch.Tensor` with dim `{input.ndim}`")

    if isinstance(other, PHETensor):
        raise NotImplementedError(
            f"rmatmul {input} with {other} not supported, phe is not multiplicative homomorphic"
        )

    if not isinstance(other, torch.Tensor):
        return NotImplemented

    evaluator = input.evaluator
    pk = input.pk
    coder = input.coder
    shape = input.shape
    device = input.device
    other_shape = other.shape
    output_dtype = torch.promote_types(input.dtype, other.dtype)
    output_shape = torch.matmul(torch.rand(*other_shape, device="meta"), torch.rand(*shape, device="meta")).shape
    data = evaluator.rmatmul(input.data, other, shape, other_shape, pk, coder, output_dtype)
    return PHETensor(pk, evaluator, coder, output_shape, data, output_dtype, device)


@implements(torch.matmul)
def matmul(input, other):
    if not isinstance(input, PHETensor) and isinstance(other, PHETensor):
        return rmatmul_f(other, input)

    if input.ndim > 2 or input.ndim < 1:
        raise ValueError(f"can't matmul `PHETensor` with `torch.Tensor` with dim `{input.ndim}`")

    if isinstance(other, PHETensor):
        raise ValueError("can't matmul `PHETensor` with `PHETensor`")

    if not isinstance(other, torch.Tensor):
        return NotImplemented

    evaluator = input.evaluator
    pk = input.pk
    coder = input.coder
    shape = input.shape
    device = input.device
    other_shape = other.shape
    output_dtype = torch.promote_types(input.dtype, other.dtype)
    output_shape = torch.matmul(torch.rand(*shape, device="meta"), torch.rand(*other_shape, device="meta")).shape
    data = evaluator.matmul(input.data, other, shape, other_shape, pk, coder, output_dtype)
    return PHETensor(pk, evaluator, coder, output_shape, data, output_dtype, device)


@implements(_custom_ops.slice_f)
def slice_f(input, item):
    evaluator = input.evaluator
    stride = input.shape[1]
    start = stride * item
    data = evaluator.slice(input._data, start, stride)
    device = input.device
    return PHETensor(input.pk, evaluator, input.coder, torch.Size([*input.shape[1:]]), data, input.dtype, device)


@implements(_custom_ops.to_local_f)
def to_local_f(input):
    return input
