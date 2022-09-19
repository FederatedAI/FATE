import typing

from ...abc.tensor import (
    PHECipherABC,
    PHEDecryptorABC,
    PHEEncryptorABC,
    PHETensorABC,
)


class Local:
    @property
    def block(self):
        ...

    def is_distributed(self):
        return False


def phe_tensor_metaclass(fp_cls):
    class PHETensorMetaclass(type):
        def __new__(cls, name, bases, dict):
            phe_cls = super().__new__(cls, name, (*bases, Local), dict)

            def __init__(self, block) -> None:
                self._block = block
                self._is_transpose = False

            setattr(phe_cls, "__init__", __init__)

            @property
            def shape(self):
                return self._block.shape

            setattr(phe_cls, "shape", shape)

            @property
            def T(self) -> phe_cls:
                transposed = phe_cls(self._block)
                transposed._is_transpose = not self._is_transpose
                return transposed

            setattr(phe_cls, "T", T)

            def serialize(self) -> bytes:
                # todo: impl me
                ...

            setattr(phe_cls, "serialize", serialize)

            def __add__(self, other):
                if isinstance(other, phe_cls):
                    other = other._block

                if isinstance(other, (phe_cls, fp_cls)):
                    return phe_cls(self._block + other)
                elif isinstance(other, (int, float)):
                    return phe_cls(self._block + other)
                else:
                    return NotImplemented

            def __radd__(self, other):
                return __add__(other, self)

            setattr(phe_cls, "__add__", __add__)
            setattr(phe_cls, "__radd__", __radd__)

            def __sub__(self, other):
                if isinstance(other, phe_cls):
                    other = other._block

                if isinstance(other, (phe_cls, fp_cls)):
                    return phe_cls(self._block - other)
                elif isinstance(other, (int, float)):
                    return phe_cls(self._block - other)
                else:
                    return NotImplemented

            def __rsub__(self, other):
                return __sub__(other, self)

            setattr(phe_cls, "__sub__", __sub__)
            setattr(phe_cls, "__rsub__", __rsub__)

            def __mul__(self, other):
                if isinstance(other, fp_cls):
                    return phe_cls(self._block * other)
                elif isinstance(other, (int, float)):
                    return phe_cls(self._block * other)
                else:
                    return NotImplemented

            def __rmul__(self, other):
                return __mul__(other, self)

            setattr(phe_cls, "__mul__", __mul__)
            setattr(phe_cls, "__rmul__", __rmul__)

            def __matmul__(self, other):
                if isinstance(other, fp_cls):
                    return phe_cls(self._block @ other)
                return NotImplemented

            def __rmatmul__(self, other):
                if isinstance(other, fp_cls):
                    return phe_cls(other @ self._block)
                return NotImplemented

            setattr(phe_cls, "__matmul__", __matmul__)
            setattr(phe_cls, "__rmatmul__", __rmatmul__)

            return phe_cls

    return PHETensorMetaclass


def phe_tensor_encryptor_metaclass(phe_cls, fp_cls):
    class PHETensorEncryptorMetaclass(type):
        def __new__(cls, name, bases, dict):
            phe_encrypt_cls = super().__new__(cls, name, bases, dict)

            def __init__(self, block_encryptor):
                self._block_encryptor = block_encryptor

            def encrypt(self, tensor: fp_cls) -> phe_cls:
                return phe_cls(self._block_encryptor.encrypt(tensor))

            setattr(phe_encrypt_cls, "__init__", __init__)
            setattr(phe_encrypt_cls, "encrypt", encrypt)
            return phe_encrypt_cls

    return PHETensorEncryptorMetaclass


def phe_tensor_decryptor_metaclass(phe_cls, fp_cls):
    class PHETensorDecryptorMetaclass(type):
        def __new__(cls, name, bases, dict):
            phe_decrypt_cls = super().__new__(cls, name, bases, dict)

            def __init__(self, block_decryptor) -> None:
                self._block_decryptor = block_decryptor

            def decrypt(self, tensor: phe_cls) -> fp_cls:
                return self._block_decryptor.decrypt(tensor._block)

            setattr(phe_decrypt_cls, "__init__", __init__)
            setattr(phe_decrypt_cls, "decrypt", decrypt)
            return phe_decrypt_cls

    return PHETensorDecryptorMetaclass


def phe_tensor_cipher_metaclass(
    phe_cls, phe_encrypt_cls, phe_decrypt_cls, block_cipher,
):
    class PHETensorCipherMetaclass(type):
        def __new__(cls, name, bases, dict):
            phe_cipher_cls = super().__new__(cls, name, bases, dict)

            @classmethod
            def keygen(cls, **kwargs) -> typing.Tuple[phe_encrypt_cls, phe_decrypt_cls]:
                block_encrytor, block_decryptor = block_cipher.keygen(**kwargs)
                return (
                    phe_encrypt_cls(block_encrytor),
                    phe_decrypt_cls(block_decryptor),
                )

            setattr(phe_cipher_cls, "keygen", keygen)
            return phe_cipher_cls

    return PHETensorCipherMetaclass
