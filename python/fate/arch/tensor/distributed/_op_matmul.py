from fate.arch.storage import storage_ops
from fate.arch.storage._shape import DAxis, Shape

from .._storage import DStorage


def matmul(a: DStorage, b: DStorage):
    bc_shape_a = a.shape[:-2]
    bc_shape_b = b.shape[:-2]
    bs_shape = Shape.broadcast_shape([bc_shape_a, bc_shape_b], raise_exception=False)
    if bs_shape is None:
        raise ValueError("matmul: shape broadcast failed")

    if bc_shape_a.d_axis is not None:
        # distributed along bc part: (...,d,...,m, k) x (...,d,...,k, n) -> (...,d,..., m, n)
        # join and matmul
        return storage_ops.matmul(a, b, [], {}, bc_shape_validate=False)

    mul_shape_a = a.shape[-2:]
    mul_shape_b = b.shape[-2:]
    if mul_shape_a.size[-1] != mul_shape_b.size[0]:
        raise ValueError("matmul: dimension mismatch: should be (..., n) x (...,n,?)")

    if mul_shape_a.is_d_axis(-2) and mul_shape_b.is_d_axis(-1):
        raise ValueError(
            f"not supported distributed axis position (...,d,?) for left tensor {a} and distributed axis position (...,?,d) for right tensor {b}"
        )

    if mul_shape_a.is_d_axis(-2) and mul_shape_b.d_axis is None:
        shape = Shape(
            size=[*bs_shape.size, mul_shape_a.size[0], mul_shape_b.size[-1]],
            d_axis=DAxis(len(bs_shape.size) + mul_shape_a.d_axis.axis, mul_shape_a.d_axis.partitions),
        )
        out_storage = DStorage.elemwise_bc_op(a, b, lambda l, r: storage_ops.matmul(l, r), shape=shape)
    elif mul_shape_b.is_d_axis(-1) and mul_shape_a.d_axis is None:
        shape = (
            Shape(
                size=[*bs_shape.size, mul_shape_a.size[0], mul_shape_b.size[-1]],
                d_axis=DAxis(len(bs_shape.size) + mul_shape_b.d_axis.axis, mul_shape_b.d_axis.partitions),
            ),
        )
        out_storage = DStorage.elemwise_bc_op(a, b, lambda l, r: storage_ops.matmul(l, r), shape=bs_shape)
    else:
        out_storage = a.blocks.join(
            b.blocks,
            _apply_transpose(
                storage_ops.matmul,
                a.transposed,
                b.transposed,
            ),
        ).reduce(storage_ops.add)
    return out_storage


def _apply_transpose(func, lflag, rflag):
    def _wrap(lblk, rblk):
        if lflag:
            lblk = lblk.transpose()
        if rflag:
            rblk = rblk.transpose()
        return func(lblk, rblk)

    return _wrap
