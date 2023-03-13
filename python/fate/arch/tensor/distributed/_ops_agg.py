import torch

from ._tensor import DTensor, implements


@implements(torch.sum)
def sum(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is not None and not kwargs.get("keepdim", False):
        kwargs["keepdim"] = False
    block_table = input.blocks.mapValues(lambda x: torch.sum(x, *args, **kwargs))
    if dim is None or dim == input._d_axis.axis:
        return block_table.reduce(torch.add)
    return DTensor(block_table, input.shape, input._d_axis, input._dtype, input._device)


@implements(torch.mean)
def mean(input: DTensor, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is not None and dim != input._d_axis.axis:
        count = input.shape[dim]
        return DTensor(
            input.blocks.mapValues(lambda x: torch.true_divide(torch.sum(x, *args, **kwargs), count)),
            input.shape,
            input._d_axis,
            input._dtype,
            input._device,
        )
    else:
        if dim is None:
            count = input.shape.numel()
        else:
            count = input.shape[dim]
        output = input.blocks.mapValues(lambda x: torch.sum(x, *args, **kwargs)).reduce(lambda x, y: torch.add(x, y))
        return torch.true_divide(output, count)


@implements(torch.min)
def min(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is None:

        def _mapper(x):
            return storage_ops.min(x)

        return storage.blocks.mapValues(_mapper).reduce(lambda x, y: storage_ops.minimum(x, y))
    else:
        if dim == storage.d_axis.axis:
            return storage.blocks.mapValues(lambda x: storage_ops.min(x, dim=dim)).reduce(
                lambda x, y: storage_ops.minimum(x, y)
            )
        else:
            return _unary_op(storage, lambda s: storage_ops.min(s, *args, **kwargs))


@implements(torch.max)
def max(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is None:

        def _mapper(x):
            return storage_ops.max(x)

        return storage.blocks.mapValues(_mapper).reduce(lambda x, y: storage_ops.maximum(x, y))
    else:
        if dim == storage.d_axis.axis:
            return storage.blocks.mapValues(lambda x: storage_ops.max(x, dim=dim)).reduce(
                lambda x, y: storage_ops.maximum(x, y)
            )
        else:
            return _unary_op(storage, lambda s: storage_ops.max(s, *args, **kwargs))


@implements(torch.std)
def std(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    unbiased = kwargs.get("unbiased", True)

    if dim is not None and dim != storage.d_axis.axis:
        return _unary_op(storage, lambda x: storage_ops.std(x, dim=dim, unbiased=unbiased))

    else:
        if dim is None:
            n = storage.shape.prod()

            def _mapper(x):
                return (storage_ops.sum(storage_ops.square(x)), storage_ops.sum(x))

        else:
            n = storage.shape[dim]

            def _mapper(x):
                return (storage_ops.sum(storage_ops.square(x), dim=dim), storage_ops.sum(x, dim=dim))

        def _reducer(x, y):
            return (storage_ops.add(x[0], y[0]), storage_ops.add(x[1], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        output = storage_ops.sub(storage_ops.div(sq, n), storage_ops.square(storage_ops.div(s, n)))
        if unbiased:
            output = storage_ops.mul(output, n / (n - 1))
        output = storage_ops.sqrt(output)
        return output


@implements(torch.var)
def var(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    unbiased = kwargs.get("unbiased", True)

    if dim is not None and dim != storage.d_axis.axis:
        return _unary_op(storage, lambda x: storage_ops.var(x, dim=dim, unbiased=unbiased))

    else:
        if dim is None:
            n = storage.shape.prod()

            def _mapper(x):
                return (storage_ops.sum(storage_ops.square(x)), storage_ops.sum(x))

        else:
            n = storage.shape[dim]

            def _mapper(x):
                return (storage_ops.sum(storage_ops.square(x), dim=dim), storage_ops.sum(x, dim=dim))

        def _reducer(x, y):
            return (storage_ops.add(x[0], y[0]), storage_ops.add(x[1], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        output = storage_ops.sub(storage_ops.div(sq, n), storage_ops.square(storage_ops.div(s, n)))
        if unbiased:
            output = storage_ops.mul(output, n / (n - 1))
        return output
