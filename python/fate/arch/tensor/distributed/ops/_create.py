from fate.arch.unify import device as D


def ones(shape, dtype, device, distributed_setting):
    ctx, partitions, d_axis = distributed_setting
    raise NotImplementedError()


def zeros(shape, dtype, device, distributed_setting):
    raise NotImplementedError()


def randn(shape, dtype, device, distributed_setting):
    raise NotImplementedError()
