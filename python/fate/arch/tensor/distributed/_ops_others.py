from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.to_local_f)
def to_local_f(input: DTensor):
    return input.shardings.merge()
