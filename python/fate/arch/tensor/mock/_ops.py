import torch
from fate.arch.tensor import _custom_ops

from ._tensor import MockPaillierTensor, implements


@implements(_custom_ops.decrypt_f)
def decrypt_f(input, decryptor):
    return decryptor.decrypt_tensor(input)


@implements(torch.add)
def add(input: MockPaillierTensor, other):
    if not isinstance(input, MockPaillierTensor) and isinstance(other, MockPaillierTensor):
        return add(other, input)

    if isinstance(other, MockPaillierTensor):
        return MockPaillierTensor(torch.add(input._data, other._data))
    if isinstance(other, torch.Tensor):
        return MockPaillierTensor(torch.add(input._data, other.detach()))
    if isinstance(other, (float, int)):
        return MockPaillierTensor(torch.add(input._data, other))
    return NotImplemented


@implements(torch.rsub)
def rsub(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, MockPaillierTensor) and isinstance(other, MockPaillierTensor):
        return sub(other, input)

    if isinstance(other, MockPaillierTensor):
        return MockPaillierTensor(torch.rsub(input._data, other._data))

    if isinstance(other, torch.Tensor):
        return MockPaillierTensor(torch.rsub(input._data, other.detach()))

    if isinstance(other, (float, int)):
        return MockPaillierTensor(torch.rsub(input._data, other))
    return NotImplemented


@implements(torch.sub)
def sub(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, MockPaillierTensor) and isinstance(other, MockPaillierTensor):
        return rsub(other, input)

    if isinstance(other, MockPaillierTensor):
        return MockPaillierTensor(torch.sub(input._data, other._data))

    if isinstance(other, torch.Tensor):
        return MockPaillierTensor(torch.sub(input._data, other.detach()))
    if isinstance(other, (float, int)):
        return MockPaillierTensor(torch.sub(input._data, other))
    return NotImplemented


@implements(torch.mul)
def mul(input, other):
    # assert input is PaillierTensor
    if not isinstance(input, MockPaillierTensor) and isinstance(other, MockPaillierTensor):
        return mul(other, input)

    if isinstance(other, MockPaillierTensor):
        raise ValueError("can't mul `PaillierTensor` with `PaillierTensor`")

    if isinstance(other, torch.Tensor):
        return MockPaillierTensor(torch.mul(input._data, other.detach()))
    if isinstance(other, (float, int)):
        return MockPaillierTensor(torch.mul(input._data, other))
    return NotImplemented


@implements(_custom_ops.rmatmul_f)
def rmatmul_f(input, other):
    if not isinstance(input, MockPaillierTensor) and isinstance(other, MockPaillierTensor):
        return matmul(other, input)

    if isinstance(other, torch.Tensor):
        return MockPaillierTensor(torch.matmul(other.detach(), input._data))
    return NotImplemented


@implements(torch.matmul)
def matmul(input, other):
    if not isinstance(input, MockPaillierTensor) and isinstance(other, MockPaillierTensor):
        return rmatmul_f(other, input)

    if isinstance(other, MockPaillierTensor):
        raise ValueError("can't matmul `PaillierTensor` with `PaillierTensor`")

    if isinstance(other, torch.Tensor):
        return MockPaillierTensor(torch.matmul(input._data, other.detach()))
    return NotImplemented


@implements(_custom_ops.to_local_f)
def to_local_f(input):
    return input
