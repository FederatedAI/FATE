import logging

import torch
from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(_custom_ops.rmatmul_f)
def rmatmul_f(a: DTensor, b: DTensor):
    assert isinstance(a, DTensor) or isinstance(b, DTensor), "atleast one dtensor"
    if not isinstance(a, DTensor):
        return matmul(b, a)

    if len(a.shape) == 1 and len(b.shape) == 1:
        if isinstance(b, DTensor):
            return a.shardings.join_reduce_shard(b.shardings, _custom_ops.rmatmul_f, torch.add)
        else:
            assert a.shape[0] == b.shape[0], f"shapes mismatch: {a.shape} and {b.shape}"
            logger.warning("matmul shape 1 distributed tensor with local shape 1 tensor maybe slow")
            return a.shardings.map_reduce_shard_with_stride(
                lambda stride, size, s: _custom_ops.rmatmul_f(s, b[stride : stride + size]), torch.add
            )

    if len(a.shape) == 1 and len(b.shape) > 1:
        if isinstance(b, DTensor):
            assert b.shardings.shapes.axis == len(b.shardings.shape) - 1, "distributed axis mismatch"
            return a.shardings.join_reduce_shard(b.shardings, _custom_ops.rmatmul_f, torch.add)
        else:
            assert a.shape[0] == b.shape[-2:][-1], f"shapes mismatch: {a.shape} and {b.shape}"
            logger.warning("rmatmul shape 1 distributed tensor with local tensor maybe slow")
            axis = len(b.shape) - 1

            def _mapper(stride, size, s):
                slices = tuple(
                    slice(stride, stride + size) if i == axis else slice(None, None, None) for i in range(len(b.shape))
                )
                return _custom_ops.rmatmul_f(s, b[slices])

            return a.shardings.map_reduce_shard_with_stride(_mapper, torch.add)

    if len(a.shape) > 1 and len(b.shape) == 1:
        if isinstance(b, DTensor):
            assert a.shardings.shapes.axis == len(a.shardings.shape) - 2, "distributed axis mismatch"
            return a.shardings.join_reduce_shard(b.shardings, _custom_ops.rmatmul_f, torch.add)
        else:
            assert a.shape[-2] == b.shape[0], f"shapes mismatch: {a.shape} and {b.shape}"
            logger.warning("matmul shape 1 distributed tensor with local tensor maybe slow")

            def _mapper(stride, size, s):
                slices = slice(stride, stride + size, 1)
                return _custom_ops.rmatmul_f(s, b[slices])

            return a.shardings.map_reduce_shard_with_stride(_mapper, torch.add)

    else:
        if isinstance(b, DTensor):
            na_axis = a.shardings.shapes.axis - len(a.shape)
            nb_axis = b.shardings.shapes.axis - len(b.shape)
            # distributed axis in broadcast part
            if na_axis < -2 and nb_axis < -2 and na_axis == nb_axis:
                shapes = [
                    torch.Size([*torch.broadcast_shapes(sa[:-2], sb[:-2]), sb[-2], sa[-1]])
                    for sa, sb in zip(a.shardings.shapes.shapes, b.shardings.shapes.shapes)
                ]
                axis = len(shapes[0]) + na_axis
                return DTensor(
                    a.shardings.join_shard(b.shardings, func=_custom_ops.rmatmul_f, out_shapes=shapes, out_axis=axis)
                )

            # distributed axis in matmul part
            elif na_axis == -2 and nb_axis == -1:
                return a.shardings.join_reduce_shard(
                    b.shardings, mapper_func=_custom_ops.rmatmul_f, reduce_func=torch.add
                )
            else:
                raise RuntimeError(f"invalid shape {a.shape} and {b.shape}")

        else:
            assert a.shape[-1] == b.shape[-2], f"shapes mismatch: {a.shape} and {b.shape}"
            na_axis = a.shardings.shapes.axis - len(a.shape)
            if na_axis != -2:
                shapes = [
                    torch.Size([*torch.broadcast_shapes(sa[:-2], b.shape[:-2]), b.shape[-2], sa[-1]])
                    for sa in a.shardings.shapes.shapes
                ]
                axis = len(shapes[0]) + na_axis
                return DTensor(a.shardings.map_shard(lambda x: _custom_ops.rmatmul_f(x, b), shapes=shapes, axis=axis))
            else:
                logger.warning("matmul shape 1 distributed tensor with local tensor maybe slow")
                axis = len(b.shape) - 1

                def _mapper(stride, size, s):
                    slices = tuple(
                        slice(stride, stride + size, 1) if i == axis else slice(None, None, None)
                        for i in range(len(b.shape))
                    )
                    return _custom_ops.rmatmul_f(s, b[slices])

                return a.shardings.map_reduce_shard_with_stride(_mapper, torch.add)


def promote_torch_matmul(a: torch.Tensor, b: torch.Tensor):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        target_dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.type(target_dtype)
        b = b.type(target_dtype)
    return torch.matmul(_maybe_detach(a), _maybe_detach(b))


def _maybe_detach(a):
    if isinstance(a, torch.Tensor):
        return a.detach()
    return a


@implements(torch.matmul)
def matmul(a: DTensor, b: DTensor):
    assert isinstance(a, DTensor) or isinstance(b, DTensor), "atleast one dtensor"
    if not isinstance(a, DTensor):
        return rmatmul_f(b, a)

    if len(a.shape) == 1 and len(b.shape) == 1:
        if isinstance(b, DTensor):
            return a.shardings.join_reduce_shard(b.shardings, promote_torch_matmul, torch.add)
        else:
            assert a.shape[0] == b.shape[0], f"shapes mismatch: {a.shape} and {b.shape}"
            logger.warning("matmul shape 1 distributed tensor with local shape 1 tensor maybe slow")
            return a.shardings.map_reduce_shard_with_stride(
                lambda stride, size, s: promote_torch_matmul(s, b[stride : stride + size]), torch.add
            )

    elif len(a.shape) == 1 and len(b.shape) > 1:
        if isinstance(b, DTensor):
            assert b.shardings.shapes.axis == len(b.shardings.shape) - 2, "distributed axis mismatch"
            return a.shardings.join_reduce_shard(b.shardings, promote_torch_matmul, torch.add)
        else:
            assert a.shape[0] == b.shape[-2:][0], f"shapes mismatch: {a.shape} and {b.shape}"
            logger.warning("matmul shape 1 distributed tensor with local tensor maybe slow")
            axis = len(b.shape) - 2

            def _mapper(stride, size, s):
                slices = tuple(
                    slice(stride, stride + size, 1) if i == axis else slice(None, None, None)
                    for i in range(len(b.shape))
                )
                return promote_torch_matmul(s, b[slices])

            return a.shardings.map_reduce_shard_with_stride(_mapper, torch.add)

    elif len(a.shape) > 1 and len(b.shape) == 1:
        if isinstance(b, DTensor):
            assert a.shardings.shapes.axis == len(a.shardings.shape) - 1, "distributed axis mismatch"
            return a.shardings.join_reduce_shard(b.shardings, promote_torch_matmul, torch.add)
        else:
            assert a.shape[-1] == b.shape[0], f"shapes mismatch: {a.shape} and {b.shape}"
            logger.warning("matmul shape 1 distributed tensor with local tensor maybe slow")

            def _mapper(stride, size, s):
                slices = slice(stride, stride + size, 1)
                return promote_torch_matmul(s, b[slices])

            return a.shardings.map_reduce_shard_with_stride(_mapper, torch.add)

    else:
        if isinstance(b, DTensor):
            na_axis = a.shardings.shapes.axis - len(a.shape)
            nb_axis = b.shardings.shapes.axis - len(b.shape)
            # distributed axis in broadcast part
            if na_axis < -2 and nb_axis < -2 and na_axis == nb_axis:
                shapes = [
                    torch.Size([*torch.broadcast_shapes(sa[:-2], sb[:-2]), sa[-2], sb[-1]])
                    for sa, sb in zip(a.shardings.shapes.shapes, b.shardings.shapes.shapes)
                ]
                axis = len(shapes[0]) + na_axis
                return DTensor(
                    a.shardings.join_shard(b.shardings, func=promote_torch_matmul, out_shapes=shapes, out_axis=axis)
                )

            # distributed axis in matmul part
            elif na_axis == -1 and nb_axis == -2:
                return a.shardings.join_reduce_shard(
                    b.shardings, mapper_func=promote_torch_matmul, reduce_func=torch.add
                )
            else:
                raise RuntimeError(f"invalid shape {a.shardings.shapes} and {b.shardings.shapes}")

        else:
            assert a.shape[-1] == b.shape[-2], f"shapes mismatch: {a.shape} and {b.shape}"
            na_axis = a.shardings.shapes.axis - len(a.shape)
            if na_axis != -1:
                shapes = [
                    torch.Size([*torch.broadcast_shapes(sa[:-2], b.shape[:-2]), sa[-2], b.shape[-1]])
                    for sa in a.shardings.shapes.shapes
                ]
                axis = len(shapes[0]) + na_axis
                return DTensor(a.shardings.map_shard(lambda x: promote_torch_matmul(x, b), shapes=shapes, axis=axis))
            else:
                logger.warning("matmul shape 1 distributed tensor with local tensor maybe slow")
                axis = len(b.shape) - 2

                def _mapper(stride, size, s):
                    slices = tuple(
                        slice(stride, stride + size, 1) if i == axis else slice(None, None, None)
                        for i in range(len(b.shape))
                    )
                    return promote_torch_matmul(s, b[slices])

                return a.shardings.map_reduce_shard_with_stride(_mapper, torch.add)
