from .._base import DStorage


def sum(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    output = DStorage.unary_op(storage, lambda x: x.sum(*args, **kwargs))
    if dim is None or dim == storage.shape.d_axis:
        from ..device import _ops_dispatch_signature2_local_unknown_unknown

        add = _ops_dispatch_signature2_local_unknown_unknown("add", storage._device, storage.dtype, [], {})

        output = output.blocks.reduce(lambda x, y: add(x, y))
    return output


def mean(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    output = DStorage.unary_op(storage, lambda x: x.sum(*args, **kwargs))

    from ..device import _ops_dispatch_signature2_local_unknown_unknown

    if dim is None or dim == storage.shape.d_axis:
        add = _ops_dispatch_signature2_local_unknown_unknown("add", storage._device, storage.dtype, [], {})

        output = output.blocks.reduce(lambda x, y: add(x, y))

    truediv = _ops_dispatch_signature2_local_unknown_unknown("true_divide", storage._device, storage.dtype, [], {})
    if dim is None:
        count = storage.shape.prod()
    else:
        count = storage.shape[dim]
    return truediv(output, count)


def max(self, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    output = self.unary_op(self, lambda x: x.sum(*args, **kwargs))
    if dim is None or dim == self.shape.d_axis:
        from .device import _ops_dispatch_signature2_local_unknown_unknown

        add = _ops_dispatch_signature2_local_unknown_unknown("add", self._device, self.dtype, [], {})

        output = output.blocks.reduce(lambda x, y: add(x, y))
    return output


def var(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]

    from ..device import (
        _ops_dispatch_signature1_local_unknown_unknown,
        _ops_dispatch_signature2_local_unknown_unknown,
    )

    _device = storage.device
    _dtype = storage.dtype
    _square = _ops_dispatch_signature1_local_unknown_unknown("square", _device, _dtype, [], {})
    _sum = _ops_dispatch_signature1_local_unknown_unknown("sum", _device, _dtype, [], {})
    _add = _ops_dispatch_signature2_local_unknown_unknown("add", _device, _dtype, [], {})
    _div = _ops_dispatch_signature2_local_unknown_unknown("div", _device, _dtype, [], {})
    _sub = _ops_dispatch_signature2_local_unknown_unknown("sub", _device, _dtype, [], {})
    if dim is None:

        def _mapper(x):
            return (_sum(_square(x)), _sum(x))

        def _reducer(x, y):
            return (_add(x[0], y[0]), _add(y[0], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        n = storage.shape.prod()
        return _sub(_square(_div(s, n)), _div(sq, n))


def std(storage: DStorage, *args, **kwargs):
    dim = None
    if len(args) > 0:
        dim = args[0]
    if "dim" in kwargs:
        dim = kwargs["dim"]

    from ..device import (
        _ops_dispatch_signature1_local_unknown_unknown,
        _ops_dispatch_signature2_local_unknown_unknown,
    )

    _device = storage.device
    _dtype = storage.dtype
    _square = _ops_dispatch_signature1_local_unknown_unknown("square", _device, _dtype, [], {})
    _sum = _ops_dispatch_signature1_local_unknown_unknown("sum", _device, _dtype, [], {})
    _add = _ops_dispatch_signature2_local_unknown_unknown("add", _device, _dtype, [], {})
    _div = _ops_dispatch_signature2_local_unknown_unknown("div", _device, _dtype, [], {})
    _sub = _ops_dispatch_signature2_local_unknown_unknown("sub", _device, _dtype, [], {})
    _sqrt = _ops_dispatch_signature1_local_unknown_unknown("sqrt", _device, _dtype, [], {})
    if dim is None:

        def _mapper(x):
            return (_sum(_square(x)), _sum(x))

        def _reducer(x, y):
            return (_add(x[0], y[0]), _add(y[0], y[1]))

        sq, s = storage.blocks.mapValues(_mapper).reduce(_reducer)
        n = storage.shape.prod()
        return _sqrt(_sub(_square(_div(s, n)), _div(sq, n)))
