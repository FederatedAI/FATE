from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.slice_f)
def slice_f(input: DTensor, key):
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
            return (ind, storage_ops.slice(s, partition_keys[ind]))

        blocks = storage.blocks.map(mapper)
        size = (len(key), *storage.shape.size[1:])
        d_axis = DAxis(axis=storage.d_axis.axis, partitions=[len(p) for p in partition_keys])

        return DStorage(
            blocks,
            shape=Shape(size, d_axis),
            dtype=storage.dtype,
            device=storage.device,
            transposed=storage.transposed,
        )
    else:
        raise NotImplementedError(f"key {key}")
