from typing import Tuple

import torch

from ._tensor import DTensor, implements


@implements(torch.sum)
def sum(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)
    dtype = kwargs.get("dtype", None)
    if dim is None:
        if "keepdim" in kwargs:
            raise TypeError(
                f"sum() received an invalid combination of arguments - got (Tensor, keepdim=bool), but expected one of\n"
                "* (Tensor input)\n"
                "* (Tensor input, tuple of ints dim, bool keepdim)\n"
                "* (Tensor input, tuple of names dim, bool keepdim)"
            )
        out = input.shardings.map_reduce_shard(lambda x: torch.sum(x), torch.add)
        if dtype:
            out = out.type(dtype)
        return out

    keepdim = kwargs.get("keepdim", False)
    if input.shardings.shapes.axis not in dim:
        return DTensor(
            input.shardings.map_shard(
                lambda x: torch.sum(x, dim=dim, keepdim=keepdim, dtype=dtype),
                shapes=input.shardings.shapes.squeeze(dim, keepdim),
            )
        )

    out = input.shardings.map_reduce_shard(lambda x: torch.sum(x, dim=dim, keepdim=keepdim), torch.add)
    if dtype:
        out = out.type(dtype)
    return out


@implements(torch.mean)
def mean(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)
    dtype = kwargs.get("dtype", None)
    if dtype is None:
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                f"mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: {input.dtype}"
            )
        dtype = input.dtype
    if dim is None:
        if "keepdim" in kwargs:
            raise TypeError(
                f"mean() received an invalid combination of arguments - got (Tensor, keepdim=bool), but expected one of\n"
                "* (Tensor input)\n"
                "* (Tensor input, tuple of ints dim, bool keepdim)\n"
                "* (Tensor input, tuple of names dim, bool keepdim)"
            )
        return torch.div(
            input.shardings.map_reduce_shard(lambda x: torch.sum(x, dtype=torch.float64), torch.add),
            input.shape.numel(),
        ).type(dtype)

    keepdim = kwargs.get("keepdim", False)
    count = 1
    for d in dim:
        count *= input.shape[d]
    if input.shardings.shapes.axis not in dim:
        return DTensor(
            input.shardings.map_shard(
                lambda x: torch.div(torch.sum(x, dim=dim, keepdim=keepdim, dtype=torch.float64), count).type(dtype),
                shapes=input.shardings.shapes.squeeze(dim, keepdim),
            )
        )

    return torch.div(
        input.shardings.map_reduce_shard(
            lambda x: torch.sum(x, dim=dim, keepdim=keepdim, dtype=torch.float64), torch.add
        ),
        count,
    ).type(dtype)


@implements(torch.std)
def std(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)
    dtype = kwargs.get("dtype", None)
    if dtype is None:
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                f"std(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: {input.dtype}"
            )
        dtype = input.dtype
    unbiased = kwargs.get("unbiased", True)
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        if "keepdim" in kwargs:
            raise TypeError(
                f"std() received an invalid combination of arguments - got (Tensor, keepdim=bool), but expected one of\n"
                "* (Tensor input)\n"
                "* (Tensor input, tuple of ints dim, bool keepdim)\n"
                "* (Tensor input, tuple of names dim, bool keepdim)"
            )

    if dim is None or input.shardings.shapes.axis in dim:
        if dim is None:
            n = input.shape.numel()
            sq, s = input.shardings.map_reduce_shard(
                mapper_func=lambda x: (torch.sum(torch.square(x)), torch.sum(x)),
                reducer_func=lambda a, b: (torch.add(a[0], b[0]), torch.add(a[1], b[1])),
            )
        else:
            n = 1
            for d in dim:
                n *= input.shape[d]
            sq, s = input.shardings.map_reduce_shard(
                mapper_func=lambda x: (
                    torch.sum(torch.square(x), dim=dim, keepdim=keepdim),
                    torch.sum(x, dim=dim, keepdim=keepdim),
                ),
                reducer_func=lambda a, b: (torch.add(a[0], b[0]), torch.add(a[1], b[1])),
            )
        output = torch.sub(torch.div(sq, n), torch.square(torch.div(s, n)))
        if unbiased:
            output = torch.mul(output, n / (n - 1))
        output = torch.sqrt(output)
        return output

    return DTensor(
        input.shardings.map_shard(
            lambda x: torch.std(x, dim=dim, unbiased=unbiased), shapes=input.shardings.shapes.squeeze(dim, keepdim)
        )
    )


@implements(torch.var)
def var(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)
    dtype = kwargs.get("dtype", None)
    if dtype is None:
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                f"var(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: {input.dtype}"
            )
        dtype = input.dtype
    unbiased = kwargs.get("unbiased", True)
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        if "keepdim" in kwargs:
            raise TypeError(
                f"var() received an invalid combination of arguments - got (Tensor, keepdim=bool), but expected one of\n"
                "* (Tensor input)\n"
                "* (Tensor input, tuple of ints dim, bool keepdim)\n"
                "* (Tensor input, tuple of names dim, bool keepdim)"
            )

    if dim is None or input.shardings.shapes.axis in dim:
        if dim is None:
            n = input.shape.numel()
            sq, s = input.shardings.map_reduce_shard(
                mapper_func=lambda x: (torch.sum(torch.square(x)), torch.sum(x)),
                reducer_func=lambda a, b: (torch.add(a[0], b[0]), torch.add(a[1], b[1])),
            )
        else:
            n = 1
            for d in dim:
                n *= input.shape[d]
            sq, s = input.shardings.map_reduce_shard(
                mapper_func=lambda x: (
                    torch.sum(torch.square(x), dim=dim, keepdim=keepdim),
                    torch.sum(x, dim=dim, keepdim=keepdim),
                ),
                reducer_func=lambda a, b: (torch.add(a[0], b[0]), torch.add(a[1], b[1])),
            )
        output = torch.sub(torch.div(sq, n), torch.square(torch.div(s, n)))
        if unbiased:
            output = torch.mul(output, n / (n - 1))
        return output

    return DTensor(
        input.shardings.map_shard(
            lambda x: torch.var(x, dim=dim, unbiased=unbiased), shapes=input.shardings.shapes.squeeze(dim, keepdim)
        )
    )


@implements(torch.min)
def min(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        if "keepdim" in kwargs:
            raise TypeError(
                f"min() received an invalid combination of arguments - got (Tensor, keepdim=bool), but expected one of\n"
                "* (Tensor input)\n"
                "* (Tensor input, tuple of ints dim, bool keepdim)\n"
                "* (Tensor input, tuple of names dim, bool keepdim)"
            )
        else:
            return input.shardings.map_reduce_shard(lambda x: torch.min(x), lambda x, y: torch.minimum(x, y))

    if input.shardings.shapes.axis == dim:

        def _mapper(stride: int, _: int, x: torch.Tensor):
            r = torch.min(x, dim=dim, keepdim=keepdim)
            return (stride, torch.return_types.min((r.values, r.indices + stride)))

        def _reducer(kv1: Tuple[int, torch.return_types.min], kv2: Tuple[int, torch.return_types.min]):
            s1, r1 = kv1
            s2, r2 = kv2
            s1, r1, s2, r2 = (s1, r1, s2, r2) if s1 < s2 else (s2, r2, s1, r1)
            values = torch.minimum(r1.values, r2.values)
            indices = torch.add(
                torch.mul(r1.indices, torch.le(r1.values, r2.values)),
                torch.mul(r2.indices, torch.gt(r1.values, r2.values)),
            )
            return (s1, torch.return_types.min((values, indices)))

        return input.shardings.map_reduce_shard_with_stride(_mapper, _reducer)[1]

    values = DTensor(
        input.shardings.map_shard(
            lambda x: torch.min(x, dim=dim, keepdim=keepdim).values,
            shapes=input.shardings.shapes.squeeze((dim,), keepdim=keepdim),
        )
    )
    indices = DTensor(
        input.shardings.map_shard(
            lambda x: torch.min(x, dim=dim, keepdim=keepdim).indices,
            shapes=input.shardings.shapes.squeeze((dim,), keepdim=keepdim),
        )
    )
    return torch.return_types.min((values, indices))


@implements(torch.max)
def max(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        if "keepdim" in kwargs:
            raise TypeError(
                f"max() received an invalid combination of arguments - got (Tensor, keepdim=bool), but expected one of\n"
                "* (Tensor input)\n"
                "* (Tensor input, tuple of ints dim, bool keepdim)\n"
                "* (Tensor input, tuple of names dim, bool keepdim)"
            )
        else:
            return input.shardings.map_reduce_shard(lambda x: torch.max(x), lambda x, y: torch.minimum(x, y))

    if input.shardings.shapes.axis == dim:

        def _mapper(stride: int, _: int, x: torch.Tensor):
            r = torch.max(x, dim=dim, keepdim=keepdim)
            return (stride, torch.return_types.max((r.values, r.indices + stride)))

        def _reducer(kv1: Tuple[int, torch.return_types.max], kv2: Tuple[int, torch.return_types.max]):
            s1, r1 = kv1
            s2, r2 = kv2
            s1, r1, s2, r2 = (s1, r1, s2, r2) if s1 < s2 else (s2, r2, s1, r1)
            values = torch.minimum(r1.values, r2.values)
            indices = torch.add(
                torch.mul(r1.indices, torch.ge(r1.values, r2.values)),
                torch.mul(r2.indices, torch.lt(r1.values, r2.values)),
            )
            return (s1, torch.return_types.max((values, indices)))

        return input.shardings.map_reduce_shard_with_stride(_mapper, _reducer)[1]

    values = DTensor(
        input.shardings.map_shard(
            lambda x: torch.max(x, dim=dim, keepdim=keepdim).values,
            shapes=input.shardings.shapes.squeeze((dim,), keepdim=keepdim),
        )
    )
    indices = DTensor(
        input.shardings.map_shard(
            lambda x: torch.max(x, dim=dim, keepdim=keepdim).indices,
            shapes=input.shardings.shapes.squeeze((dim,), keepdim=keepdim),
        )
    )
    return torch.return_types.max((values, indices))
