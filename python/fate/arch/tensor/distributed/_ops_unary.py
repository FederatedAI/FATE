import torch

from ._tensor import DTensor, implements


@implements(torch.exp)
def exp(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.exp, dtype_promote_to=torch.float32))


@implements(torch.log)
def log(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.log, dtype_promote_to=torch.float32))


@implements(torch.square)
def square(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.square))


@implements(torch.sigmoid)
def sigmoid(input: DTensor):
    return DTensor(input.shardings.map_shard(torch.sigmoid))
