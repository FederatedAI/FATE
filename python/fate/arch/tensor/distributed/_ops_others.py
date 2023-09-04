from fate.arch.tensor import _custom_ops
import torch

from ._tensor import DTensor, implements


@implements(_custom_ops.to_local_f)
def to_local_f(input: DTensor):
    return input.shardings.merge()


@implements(_custom_ops.encode_as_int_f)
def encode_as_int_f(input: DTensor, precision):
    return input.shardings.map_shard(lambda x: (x * 2**precision).astype(torch.int64), dtype=torch.int64)
