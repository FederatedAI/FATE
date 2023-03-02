from fate.arch.storage import dtype, storage_ops

from .._storage import DStorage
from ._dispatch import _register


def _apply_transpose(func, flag):
    def _wrap(blk):
        if flag:
            blk = blk.transpose()
        return func(blk)

    return _wrap


@_register
def encrypt(storage, encryptor):
    mapper = _apply_transpose(lambda s: storage_ops.encrypt(s, encryptor), storage.transposed)
    output_block = storage.blocks.mapValues(mapper)
    return DStorage(output_block, storage.shape, dtype.paillier, storage._device)


@_register
def decrypt(storage, decryptor):
    mapper = _apply_transpose(lambda s: storage_ops.decrypt(s, decryptor), storage.transposed)
    output_block = storage.blocks.mapValues(mapper)
    return DStorage(output_block, storage.shape, dtype.float64, storage._device)
