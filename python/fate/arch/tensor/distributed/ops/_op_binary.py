from typing import Any, Callable

from fate.arch.storage import storage_ops

from .._storage import DStorage


def elemwise_binary_op(
    a: DStorage,
    b: DStorage,
    binary_mapper: Callable[[Any, Any], Any],
    output_dtype=None,
):
    def _apply_transpose(func, lf, rf):
        def _wrap(lblk, rblk):
            if lf:
                lblk = lblk.transpose()
            if rf:
                rblk = rblk.transpose()
            return func(lblk, rblk)

        return _wrap

    binary_mapper = _apply_transpose(binary_mapper, a.transposed, b.transposed)
    output_blocks = a.blocks.join(b.blocks, binary_mapper)
    if output_dtype is None:
        output_dtype = a._dtype
    return DStorage(output_blocks, a.shape, output_dtype, a._device)


def elemwise_bc_op(
    a: "DStorage",
    b: "DStorage",
    func,
    output_dtype=None,
    shape=None,
    **kwargs,
):
    if isinstance(a, DStorage) and not isinstance(b, DStorage):
        func = _apply_transpose(func, a.transposed, False)
        output_blocks = a.blocks.mapValues(lambda x: func(x, b, **kwargs))
        if output_dtype is None:
            output_dtype = a._dtype
        if shape is None:
            shape = a.shape
        _device = a.device
    elif isinstance(b, DStorage) and not isinstance(a, DStorage):
        func = _apply_transpose(func, False, b.transposed)
        output_blocks = b.blocks.mapValues(lambda x: func(a, x, **kwargs))
        if output_dtype is None:
            output_dtype = b._dtype
        if shape is None:
            shape = b.shape
        _device = b.device
    else:
        raise RuntimeError("exactly one DStorage required")
    return DStorage(output_blocks, shape, output_dtype, _device)


def _apply_transpose(func, lf, rf):
    def _wrap(lblk, rblk):
        if lf:
            lblk = lblk.transpose()
        if rf:
            rblk = rblk.transpose()
        return func(lblk, rblk)

    return _wrap


def _binary(a, b, f):
    if isinstance(a, DStorage):
        if isinstance(b, DStorage):
            f = _apply_transpose(f, a.transposed, b.transposed)
            output_blocks = a.blocks.join(b.blocks, f)
            output_dtype = a._dtype
            return DStorage(output_blocks, a.shape, output_dtype, a._device)
        else:
            f = _apply_transpose(f, a.transposed, False)
            return DStorage(a.blocks.mapValues(lambda x: f(x, b)), a.shape, a.dtype, a.device)
    else:
        if isinstance(b, DStorage):
            f = _apply_transpose(f, False, b.transposed)
            return DStorage(b.blocks.mapValues(lambda x: f(a, x)), b.shape, b.dtype, b.device)
        else:
            raise NotImplementedError()


def mul(a, b):
    return _binary(a, b, storage_ops.mul)


def sub(a, b):
    return _binary(a, b, storage_ops.mul)


def add(a, b):
    return _binary(a, b, storage_ops.mul)


def div(a, b):
    return _binary(a, b, storage_ops.mul)
