from fate.arch.tensor.types import DStorage


def sum(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    local_ops = storage.local_ops_helper()
    output = DStorage.unary_op(storage, lambda x: local_ops.sum(x, *args, **kwargs))
    if dim is None or dim == storage.shape.d_axis:
        output = output.blocks.reduce(lambda x, y: local_ops.add(x, y))
    return output


def mean(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    local_ops = storage.local_ops_helper()
    if dim is not None and dim != storage.shape.d_axis:
        count = storage.shape[dim]
        output = DStorage.unary_op(storage, lambda x: local_ops.truediv(local_ops.sum(x, *args, **kwargs), count))
    else:
        output = DStorage.unary_op(storage, lambda x: local_ops.sum(x, *args, **kwargs))
        if dim is None:
            count = storage.shape.prod()
        else:
            count = storage.shape[dim]
        output = output.blocks.reduce(lambda x, y: local_ops.add(x, y))
        return local_ops.truediv(output, count)


def var(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    unbiased = kwargs.get("unbiased", True)

    local_ops = storage.local_ops_helper()
    if dim is not None and dim != storage.shape.d_axis:
        return DStorage.unary_op(storage, lambda x: local_ops.var(x, dim=dim, unbiased=unbiased))

    else:
        if dim is None:
            n = storage.shape.prod()

            def _mapper(x):
                return (local_ops.sum(local_ops.square(x)), local_ops.sum(x))

        else:
            n = storage.shape[dim]

            def _mapper(x):
                return (local_ops.sum(local_ops.square(x), dim=dim), local_ops.sum(x, dim=dim))

        def _reducer(x, y):
            return (local_ops.add(x[0], y[0]), local_ops.add(x[1], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        output = local_ops.sub(local_ops.div(sq, n), local_ops.square(local_ops.div(s, n)))
    if unbiased:
        output = local_ops.mul(output, n / (n - 1))
    return output


def std(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]

    local_ops = storage.local_ops_helper()
    if dim is None:

        def _mapper(x):
            return (local_ops.sum(local_ops.square(x)), local_ops.sum(x))

        def _reducer(x, y):
            return (local_ops.add(x[0], y[0]), local_ops.add(y[0], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        n = storage.shape.prod()
        return local_ops.sqrt(local_ops.sub(local_ops.square(local_ops.div(s, n)), local_ops.div(sq, n)))
    raise NotImplementedError()
