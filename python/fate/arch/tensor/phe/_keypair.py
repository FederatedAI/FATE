import typing

import torch

if typing.TYPE_CHECKING:
    from fate.arch.protocol.phe.paillier import PK, SK, Coder

    from ._tensor import PHETensor, PHETensorEncoded


class PHETensorCipher:
    def __init__(self, pk: "PHETensorEncryptor", coder: "PHETensorCoder", sk: "PHETensorDecryptor", evaluator) -> None:
        self._pk = pk
        self._coder = coder
        self._sk = sk
        self._evaluator = evaluator

    @classmethod
    def from_raw_cipher(cls, pk: "PK", coder: "Coder", sk: "SK", evaluator):
        coder = PHETensorCoder(coder)
        encryptor = PHETensorEncryptor(pk, coder, evaluator)
        decryptor = PHETensorDecryptor(sk, coder)
        return cls(encryptor, coder, decryptor, evaluator)

    @property
    def pk(self):
        return self._pk

    @property
    def coder(self):
        return self._coder

    @property
    def sk(self):
        return self._sk


class PHETensorCoder:
    def __init__(self, coder: "Coder") -> None:
        self._coder = coder

    def encode(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            from ._tensor import PHETensorEncoded

            shape = tensor.shape
            dtype = tensor.dtype
            data = self._coder.encode_tensor(tensor, dtype)
            return PHETensorEncoded(self._coder, shape, data, tensor.dtype)
        elif hasattr(tensor, "encode"):
            return tensor.encode(self)
        else:
            raise NotImplementedError(f"`{tensor}` not supported")

    def decode(self, tensor: "PHETensorEncoded"):
        from ._tensor import PHETensorEncoded

        if isinstance(tensor, PHETensorEncoded):
            return self._coder.decode_tensor(tensor.data, tensor.dtype, tensor.shape)
        elif hasattr(tensor, "decode"):
            return tensor.decode(self)
        else:
            raise NotImplementedError(f"`{tensor}` not supported")


class PHETensorEncryptor:
    def __init__(self, pk: "PK", coder: "PHETensorCoder", evaluator) -> None:
        self._pk = pk
        self._coder = coder
        self._evaluator = evaluator

    def encrypt_encoded(self, tensor: "PHETensorEncoded", obfuscate=False):
        from ._tensor import PHETensor, PHETensorEncoded

        if isinstance(tensor, PHETensorEncoded):
            data = self._pk.encrypt_encoded(tensor.data, obfuscate)
            return PHETensor(self._pk, self._evaluator, tensor.coder, tensor.shape, data, tensor.dtype)
        elif hasattr(tensor, "encrypt_encoded"):
            return tensor.encrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def encrypt_tensor(self, tensor: torch.Tensor, obfuscate=False):
        coded = self._coder.encode(tensor)
        return self.encrypt_encoded(coded, obfuscate)


class PHETensorDecryptor:
    def __init__(self, sk: "SK", coder: "PHETensorCoder") -> None:
        self._sk = sk
        self._coder = coder

    def decrypt_encoded(self, tensor: "PHETensor"):
        from ._tensor import PHETensor, PHETensorEncoded

        if isinstance(tensor, PHETensor):
            data = self._sk.decrypt_to_encoded(tensor.data)
            return PHETensorEncoded(tensor.coder, tensor.shape, data, tensor.dtype)

        elif hasattr(tensor, "decrypt_encoded"):
            return tensor.decrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def decrypt_tensor(self, tensor: "PHETensor"):
        return self._coder.decode(self.decrypt_encoded(tensor))