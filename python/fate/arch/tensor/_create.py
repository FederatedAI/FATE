import torch

from ..unify import device as D
from ._tensor import Tensor


def from_torch(t: torch.Tensor):
    from ..storage._create import from_torch

    return Tensor(from_torch(t))


# distributed only
def from_blocks(ctx, blocks, d_axis, partitions):
    from .distributed import DStorage

    storage = DStorage.from_storages(ctx, blocks, d_axis, partitions)
    return Tensor(storage)


def randn(shape, dtype, device=D.CPU, distributed_setting=None):
    if distributed_setting is None:
        from ..storage._create import randn

        return Tensor(randn(shape, dtype, device))
    else:
        from .distributed.ops._create import randn

        return Tensor(randn(shape, dtype, device, distributed_setting))


def ones(shape, dtype, device=D.CPU, distributed_setting=None):
    if distributed_setting is None:
        from ..storage._create import ones

        return Tensor(ones(shape, dtype, device))
    else:
        from .distributed.ops._create import ones

        return Tensor(ones(shape, dtype, device, distributed_setting))


def zeros(shape, dtype, device=D.CPU, distributed_setting=None):
    if distributed_setting is None:
        from ..storage._create import zeros

        return Tensor(zeros(shape, dtype, device))
    else:
        from .distributed.ops._create import zeros

        return Tensor(zeros(shape, dtype, device, distributed_setting))
