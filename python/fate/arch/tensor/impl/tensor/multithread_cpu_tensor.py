import torch

from ..blocks.multithread_cpu_paillier_block import BlockPaillierCipher
from ._metaclass import (
    phe_tensor_cipher_metaclass,
    phe_tensor_decryptor_metaclass,
    phe_tensor_encryptor_metaclass,
    phe_tensor_metaclass,
)

FPTensorLocal = torch.Tensor


class PHETensorLocal(metaclass=phe_tensor_metaclass(FPTensorLocal)):
    ...


class PaillierPHEEncryptorLocal(
    metaclass=phe_tensor_encryptor_metaclass(PHETensorLocal, FPTensorLocal)
):
    ...


class PaillierPHEDecryptorLocal(
    metaclass=phe_tensor_decryptor_metaclass(PHETensorLocal, FPTensorLocal)
):
    ...


class PaillierPHECipherLocal(
    metaclass=phe_tensor_cipher_metaclass(
        PHETensorLocal,
        PaillierPHEEncryptorLocal,
        PaillierPHEDecryptorLocal,
        BlockPaillierCipher,
    )
):
    ...
