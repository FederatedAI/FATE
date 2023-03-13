from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.encrypt)
def encrypt(input: DTensor, encryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encrypt(x, encryptor), input.dtype))


@implements(_custom_ops.decrypt)
def decrypt(input: DTensor, decryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decrypt(x, decryptor), input.dtype))
