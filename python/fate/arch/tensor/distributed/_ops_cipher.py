from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.encrypt_encoded_f)
def encrypt_encoded_f(input: DTensor, encryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encrypt_encoded_f(x, encryptor), type="encrypted"))


@implements(_custom_ops.decrypt_encoded_f)
def decrypt_encoded_f(input: DTensor, decryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decrypt_encoded_f(x, decryptor), type="encoded"))


@implements(_custom_ops.encrypt_f)
def encrypt_f(input: DTensor, encryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encrypt_f(x, encryptor), type="encrypted"))


@implements(_custom_ops.decrypt_f)
def decrypt_f(input: DTensor, decryptor):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decrypt_f(x, decryptor), type="plain"))


@implements(_custom_ops.decode_f)
def decode_f(input: DTensor, coder):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.decode_f(x, coder), type="plain"))


@implements(_custom_ops.encode_f)
def encode_f(input: DTensor, coder):
    return DTensor(input.shardings.map_shard(lambda x: _custom_ops.encode_f(x, coder), type="encoded"))
