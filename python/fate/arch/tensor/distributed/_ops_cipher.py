from fate.arch.tensor import _custom_ops

from ._tensor import DTensor, implements


@implements(_custom_ops.encrypt)
def encrypt(input: DTensor, encryptor):
    return DTensor(
        input.blocks.mapValues(lambda x: _custom_ops.encrypt(x, encryptor)),
        input.shape,
        input.d_axis,
        input.dtype,
        input.device,
    )


@implements(_custom_ops.decrypt)
def decrypt(input: DTensor, decryptor):
    return DTensor(
        input.blocks.mapValues(lambda x: _custom_ops.decrypt(x, decryptor)),
        input.shape,
        input.d_axis,
        input.dtype,
        input.device,
    )
