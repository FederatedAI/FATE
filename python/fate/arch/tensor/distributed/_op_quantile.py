from fate.arch.storage import storage_ops

from ._dispatch import _register


@_register
def quantile(storage, q, epsilon):
    summaries = storage.blocks.mapValues(lambda x: storage_ops.quantile_summary(x, epsilon)).reduce(
        lambda xl, yl: [x.merge(y) for x, y in zip(xl, yl)]
    )
    return [summary.quantile(q) for summary in summaries]
