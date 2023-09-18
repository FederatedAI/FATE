import torch

from ._tensor import DTensor, implements


@implements(torch.add)
def add(input, other):
    return _binary(input, other, torch.add)


@implements(torch.sub)
def sub(input, other):
    return _binary(input, other, torch.sub)


@implements(torch.rsub)
def rsub(input, other):
    return _binary(input, other, torch.rsub)


@implements(torch.mul)
def mul(input, other):
    return _binary(input, other, torch.mul)


@implements(torch.div)
def div(input, other):
    return _binary(input, other, torch.div, dtype_promote_to=torch.float32)


def _binary(input, other, op, swap_operad=False, dtype_promote_to=None):
    # swap input and output if input is not DTensor
    if not isinstance(input, DTensor):
        return _binary(other, input, op, swap_operad=not swap_operad, dtype_promote_to=dtype_promote_to)

    if isinstance(other, DTensor):
        if swap_operad:
            return DTensor(other.shardings.join_shard(input.shardings, op, dtype_promote_to=dtype_promote_to))
        else:
            return DTensor(input.shardings.join_shard(other.shardings, op, dtype_promote_to=dtype_promote_to))

    # other is local tensor, broadcast to partitions
    else:
        if isinstance(other, torch.Tensor):
            shapes = input.shardings.shapes.bc_shapes(other.shape)
        else:
            # other is scalar
            shapes = input.shardings.shapes.bc_shapes(torch.Size([]))

        if swap_operad:
            return DTensor(
                input.shardings.map_shard(
                    lambda x: op(other, x), dtype_promote_to=dtype_promote_to, shapes=shapes.shapes, axis=shapes.axis
                )
            )

        else:
            return DTensor(
                input.shardings.map_shard(
                    lambda x: op(x, other), dtype_promote_to=dtype_promote_to, shapes=shapes.shapes, axis=shapes.axis
                )
            )
