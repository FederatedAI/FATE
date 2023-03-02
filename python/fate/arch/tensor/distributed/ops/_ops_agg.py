from fate.arch.storage import storage_ops

from .._storage import DStorage


def _unary_op(
    a: DStorage,
    mapper,
    output_shape=None,
    output_dtype=None,
):
    def _apply_transpose(func, flag):
        def _wrap(blk):
            if flag:
                blk = blk.transpose()
            return func(blk)

        return _wrap

    mapper = _apply_transpose(mapper, a.transposed)
    output_block = a.blocks.mapValues(mapper)
    if output_dtype is None:
        output_dtype = a._dtype
    if output_shape is None:
        output_shape = a.shape
    return DStorage(output_block, output_shape, output_dtype, a._device)


def sum(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is not None and not kwargs.get("keepdim", False):
        kwargs["keepdim"] = True
    output = _unary_op(storage, lambda x: storage_ops.sum(x, *args, **kwargs))
    if dim is None or dim == storage.d_axis.axis:
        output = output.blocks.reduce(lambda x, y: storage_ops.add(x, y))
    return output


def mean(storage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    if dim is not None and dim != storage.d_axis.axis:
        count = storage.shape[dim]
        return _unary_op(storage, lambda x: storage_ops.truediv(storage_ops.sum(x, *args, **kwargs), count))
    else:
        output = _unary_op(storage, lambda x: storage_ops.sum(x, *args, **kwargs))
        if dim is None:
            count = storage.shape.prod()
        else:
            count = storage.shape[dim]
        output = output.blocks.reduce(lambda x, y: storage_ops.add(x, y))
        return storage_ops.truediv(output, count)


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
