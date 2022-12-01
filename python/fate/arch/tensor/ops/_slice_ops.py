from .._tensor import Tensor
from ..types import DStorage
from ._ops import _get_dispatch_info


def slice(a: Tensor, key) -> Tensor:
    _is_distributed, _device, _dtype = _get_dispatch_info([a])
    from ..storage._helper import local_ops_helper

    local_ops = local_ops_helper(_device, _dtype)
    if not _is_distributed:
        output_storage = local_ops.slice(a.storage)
    else:
        storage = a.storage
        assert isinstance(storage, DStorage), ""
        output_storage = DStorage.unary_op(
            storage,
            lambda s: local_ops.slice(s, key),
        )

    return Tensor(output_storage)
