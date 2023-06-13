from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.encrypt_f)
def encrypt_f(input: DTensor, encryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encrypt_f(x, encryptor)))


@implements(_custom_ops.decrypt_f)
def decrypt_f(input: DTensor, decryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decrypt_f(x, decryptor)))
