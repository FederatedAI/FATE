import typing

import torch

if typing.TYPE_CHECKING:
    from fate.arch.protocol.paillier import PK, SK, Coder

    from ._tensor import PaillierTensor, PaillierTensorEncoded


class PaillierTensorCipher:
    def __init__(
        self, pk: "PaillierTensorEncryptor", coder: "PaillierTensorCoder", sk: "PaillierTensorDecryptor"
    ) -> None:
        self._pk = pk
        self._coder = coder
        self._sk = sk

    @classmethod
    def from_raw_cipher(cls, pk: "PK", coder: "Coder", sk: "SK"):
        coder = PaillierTensorCoder(coder)
        encryptor = PaillierTensorEncryptor(pk, coder)
        decryptor = PaillierTensorDecryptor(sk, coder)
        return cls(encryptor, coder, decryptor)

    @property
    def pk(self):
        return self._pk

    @property
    def coder(self):
        return self._coder

    @property
    def sk(self):
        return self._sk


class PaillierTensorCoder:
    def __init__(self, coder: "Coder") -> None:
        self._coder = coder

    @property
    def raw_coder(self):
        return self._coder

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
    def __init__(self, pk: "PK", coder: "PaillierTensorCoder") -> None:
        self._pk = pk
        self._coder = coder

    @property
    def raw_pk(self):
        return self._pk

    def encrypt_encoded(self, tensor: "PaillierTensorEncoded", obfuscate=False):
        from ._tensor import PaillierTensor, PaillierTensorEncoded

        if isinstance(tensor, PaillierTensorEncoded):
            data = self._pk.encrypt_encoded(tensor.data, obfuscate)
            return PaillierTensor(self._pk, tensor.coder, tensor.shape, data, tensor.dtype)
        elif hasattr(tensor, "encrypt_encoded"):
            return tensor.encrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def encrypt_tensor(self, tensor: torch.Tensor, obfuscate=False):
        return self.encrypt_encoded(self._coder.encode(tensor), obfuscate)


class PaillierTensorDecryptor:
    def __init__(self, sk: "SK", coder: "PaillierTensorCoder") -> None:
        self._sk = sk
        self._coder = coder

    @property
    def raw_sk(self):
        return self._sk

    def decrypt_encoded(self, tensor: "PaillierTensor"):
        from ._tensor import PaillierTensor, PaillierTensorEncoded

        if isinstance(tensor, PaillierTensor):
            data = self._sk.decrypt_to_encoded(tensor.data)
            return PaillierTensorEncoded(tensor.coder, tensor.shape, data, tensor.dtype)

        elif hasattr(tensor, "decrypt_encoded"):
            return tensor.decrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def decrypt_tensor(self, tensor: "PaillierTensor"):
        return self._coder.decode(self.decrypt_encoded(tensor))
