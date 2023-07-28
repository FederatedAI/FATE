import typing

import torch

if typing.TYPE_CHECKING:
    from fate.arch.protocol.paillier import PK, SK, Coder

    from ._tensor import PaillierTensor, PaillierTensorEncoded


class PaillierCipher:
    def __init__(self, pk: "PK", coder: "Coder", sk: "SK") -> None:
        self._pk = pk
        self._coder = coder
        self._sk = sk

    @classmethod
    def keygen(cls, key_length):
        from fate.arch.protocol.paillier import keygen as _keygen

        sk, pk, coder = _keygen(key_length)
        return cls(pk, coder, sk)

    @property
    def pk(self):
        return PaillierTensorEncryptor(self._pk)

    @property
    def coder(self):
        return PaillierTensorCoder(self._coder)

    @property
    def sk(self):
        return PaillierTensorDecryptor(self._sk)


class PaillierTensorCoder:
    def __init__(self, coder: "Coder") -> None:
        self._coder = coder

    def encode(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            from ._tensor import PaillierTensorEncoded

            shape = tensor.shape
            tensor = tensor.flatten()
            if tensor.dtype == torch.float64:
                data = self._coder.encode_f64_vec(tensor)
            elif tensor.dtype == torch.float32:
                data = self._coder.encode_f32_vec(tensor)
            elif tensor.dtype == torch.int64:
                data = self._coder.encode_i64_vec(tensor)
            elif tensor.dtype == torch.int32:
                data = self._coder.encode_i32_vec(tensor)
            else:
                raise NotImplementedError(f"{tensor.dtype} not supported")
            return PaillierTensorEncoded(self._coder, shape, data, tensor.dtype)
        elif hasattr(tensor, "encode"):
            return tensor.encode(self)
        else:
            raise NotImplementedError(f"`{tensor}` not supported")

    def decode(self, tensor: "PaillierTensorEncoded"):
        from ._tensor import PaillierTensorEncoded

        if isinstance(tensor, PaillierTensorEncoded):
            return self._coder.decode_vec(tensor.data, tensor.dtype).reshape(tensor.shape)
        elif hasattr(tensor, "decode"):
            return tensor.decode(self)
        else:
            raise NotImplementedError(f"`{tensor}` not supported")


class PaillierTensorEncryptor:
    def __init__(self, pk: "PK") -> None:
        self._pk = pk

    def encrypt(self, tensor: "PaillierTensorEncoded", obfuscate=False):
        from ._tensor import PaillierTensor, PaillierTensorEncoded

        if isinstance(tensor, PaillierTensorEncoded):
            data = self._pk.encrypt_vec(tensor.data, obfuscate)
            return PaillierTensor(self._pk, tensor.coder, tensor.shape, data, tensor.dtype)
        elif hasattr(tensor, "encrypt"):
            return tensor.encrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")


class PaillierTensorDecryptor:
    def __init__(self, sk: "SK") -> None:
        self._sk = sk

    def decrypt(self, tensor: "PaillierTensor"):
        from ._tensor import PaillierTensor, PaillierTensorEncoded

        if isinstance(tensor, PaillierTensor):
            data = self._sk.decrypt_vec(tensor.data)
            return PaillierTensorEncoded(tensor.coder, tensor.shape, data, tensor.dtype)

        elif hasattr(tensor, "decrypt"):
            return tensor.decrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")
