from .._tensor import Tensor
from ..types import DAxis, DStorage, Shape
from ._ops import _get_dispatch_info


def slice(a: Tensor, key) -> Tensor:
    _is_distributed, _device, _dtype = _get_dispatch_info([a])
    from ..storage._helper import local_ops_helper

    local_ops = local_ops_helper(_device, _dtype)
    if not _is_distributed:
        output_storage = local_ops.slice(a.storage, key)
    else:
        storage = a.storage
        assert isinstance(storage, DStorage), ""

        if isinstance(key, list):
            partition_keys = [[] for _ in storage.d_axis.partitions]
            agg = 0
            i = 0
            j = 0
            while j < len(key) and i < len(storage.d_axis.partitions):
                if key[j] >= agg and key[j] < agg + storage.d_axis.partitions[i]:
                    partition_keys[i].append(key[j] - agg)
                    j += 1
                else:
                    agg += storage.d_axis.partitions[i]
                    i += 1
            if j != len(key):
                raise ValueError(f"out of bound: {key}")

            def mapper(ind, s):
                return (ind, local_ops.slice(s, partition_keys[ind]))

            blocks = storage.blocks.map(mapper)
            size = (len(key), *storage.shape.size[1:])
            d_axis = DAxis(axis=storage.d_axis.axis, partitions=[len(p) for p in partition_keys])

            output_storage = DStorage(
                blocks,
                shape=Shape(size, d_axis),
                dtype=storage.dtype,
                device=storage.device,
                transposed=storage.transposed,
            )
        else:
            raise NotImplementedError(f"key {key}")

    return Tensor(output_storage)
