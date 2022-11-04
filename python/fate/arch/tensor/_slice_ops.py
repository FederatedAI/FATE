from ._tensor import Tensor
from ._base import DStorage
from ._ops import _get_dispatch_info
from ._storage_ops import _ops_dispatch_signature1_local_unknown_unknown


def slice(a: Tensor, key) -> Tensor:
    _is_distributed, _device, _dtype = _get_dispatch_info([a])
    if not _is_distributed:
        storage_op = _ops_dispatch_signature1_local_unknown_unknown(
            "slice", _device, _dtype, [key], {}
        )
        output_storage = storage_op(a.storage)
    else:
        storage = a.storage
        assert isinstance(storage, DStorage), ""
        output_storage = DStorage.unary_op(
            storage,
            lambda s: _ops_dispatch_signature1_local_unknown_unknown(
                "slice", _device, _dtype, [key], {}
            )(s),
        )

    return Tensor(output_storage)
