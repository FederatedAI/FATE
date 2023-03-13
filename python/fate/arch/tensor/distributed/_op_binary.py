import torch

from ._tensor import DTensor, implements


@implements(torch.add)
def add(input, other):
    return _binary(input, other, torch.add)


@implements(torch.sub)
def sub(input, other):
    return _binary(input, other, torch.sub)


@implements(torch.mul)
def mul(input, other):
    return _binary(input, other, torch.mul)


@implements(torch.div)
def div(input, other):
    return _binary(input, other, torch.div)


def _binary(input, other, op, swap=False):
    # swap input and output if input is not DStroage
    if not isinstance(input, DTensor):
        return _binary(op, other, input, swap=not swap)

    # input and other both DStorage
    # TODO: validate
    if isinstance(other, DTensor):
        if swap:
            output_blocks = other.blocks.join(input.blocks, op)
        else:
            output_blocks = input.blocks.join(other.blocks, op)
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        output_shape = torch.broadcast_shapes(input.shape, other.shape)
        return DTensor(output_blocks, output_shape, input._d_axis, output_dtype, input._device)

    # other is local tensor, broadcast to partitions
    # TODO: validate broadcast
    else:
        if swap:
            output_blocks = input.blocks.mapValues(lambda x: op(other, x))
        else:
            output_blocks = input.blocks.mapValues(lambda x: op(x, other))
        return DTensor(output_blocks, input.shape, input._d_axis, input._dtype, input.device)
