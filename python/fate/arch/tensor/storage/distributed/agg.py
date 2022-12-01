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
    output = DStorage.unary_op(storage, lambda x: local_ops.sum(x, *args, **kwargs))

    if dim is None or dim == storage.shape.d_axis:
        output = output.blocks.reduce(lambda x, y: local_ops.add(x, y))

    if dim is None:
        count = storage.shape.prod()
    else:
        count = storage.shape[dim]
    return local_ops.truediv(output, count)


def var(storage: DStorage, *args, **kwargs):
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
        return local_ops.sub(local_ops.square(local_ops.div(s, n)), local_ops.div(sq, n))


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
