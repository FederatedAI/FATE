import torch

from ._tensor import PaillierTensor, PaillierTensorEncoded


class PaillierCipher:
    def __init__(self, pk, coder, sk) -> None:
        self._pk = pk
        self._coder = coder
        self._sk = sk

    @classmethod
    def keygen(cls, key_length):
        import fate_utils

        encryptor, decryptor = fate_utils.tensor.keygen(key_length)

        return cls(encryptor, None, decryptor)

    @property
    def pk(self):
        return PaillierTensorEncryptor(self._pk)

    @property
    def coder(self):
        return PaillierTensorCoder(self._coder)

    @property
    def sk(self):
        return PaillierTensorDecryptor(self._sk)


"""
fake PaillierTensorCoder
"""


class PaillierTensorCoder:
    def __init__(self, coder) -> None:
        ...

    def encode(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            from ._tensor import PaillierTensorEncoded

            return PaillierTensorEncoded(tensor.detach())
        elif hasattr(tensor, "encode"):
            return tensor.encode(self)
        else:
            raise NotImplementedError(f"`{tensor}` not supported")

    def decode(self, tensor: "PaillierTensorEncoded"):
        from ._tensor import PaillierTensorEncoded

        if isinstance(tensor, PaillierTensorEncoded):
            return tensor._data
        elif hasattr(tensor, "decode"):
            return tensor.decode(self)


class PaillierTensorEncryptor:
    def __init__(self, key) -> None:
        self._key = key

    def encrypt(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.float64:
                return PaillierTensor(self._key.encrypt_f64(tensor.detach().numpy()), tensor.dtype)
            if tensor.dtype == torch.float32:
                return PaillierTensor(self._key.encrypt_f32(tensor.detach().numpy()), tensor.dtype)
            if tensor.dtype == torch.int64:
                return PaillierTensor(self._key.encrypt_i64(tensor.detach().numpy()), tensor.dtype)
            if tensor.dtype == torch.int32:
                return PaillierTensor(self._key.encrypt_i32(tensor.detach().numpy()), tensor.dtype)
        elif hasattr(tensor, "encrypt"):
            return tensor.encrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def encrypt_encoded(self, tensor: "PaillierTensorEncoded"):
        from ._tensor import PaillierTensorEncoded

        if isinstance(tensor, PaillierTensorEncoded):
            if tensor._data.dtype == torch.float64:
                return PaillierTensor(self._key.encrypt_f64(tensor._data.numpy()), tensor._data.dtype)
            if tensor._data.dtype == torch.float32:
                return PaillierTensor(self._key.encrypt_f32(tensor._data.numpy()), tensor._data.dtype)
            if tensor._data.dtype == torch.int64:
                return PaillierTensor(self._key.encrypt_i64(tensor._data.numpy()), tensor._data.dtype)
            if tensor._data.dtype == torch.int32:
                return PaillierTensor(self._key.encrypt_i32(tensor._data.numpy()), tensor._data.dtype)
        elif hasattr(tensor, "encrypt_encoded"):
            return tensor.encrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")


class PaillierTensorDecryptor:
    def __init__(self, key) -> None:
        self._key = key

    def decrypt_encoded(self, tensor: PaillierTensor):
        if isinstance(tensor, PaillierTensor):
            from ._tensor import PaillierTensorEncoded

            if tensor.dtype == torch.float64:
                return PaillierTensorEncoded(torch.from_numpy(self._key.decrypt_f64(tensor._data)))
            if tensor.dtype == torch.float32:
                return PaillierTensorEncoded(torch.from_numpy(self._key.decrypt_f32(tensor._data)))
            if tensor.dtype == torch.int64:
                return PaillierTensorEncoded(torch.from_numpy(self._key.decrypt_i64(tensor._data)))
            if tensor.dtype == torch.int32:
                return PaillierTensorEncoded(torch.from_numpy(self._key.decrypt_i32(tensor._data)))
        elif hasattr(tensor, "decrypt_encoded"):
            return tensor.decrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def decrypt(self, tensor: PaillierTensor):
        if isinstance(tensor, PaillierTensor):
            if tensor.dtype == torch.float64:
                return torch.from_numpy(self._key.decrypt_f64(tensor._data))
            if tensor.dtype == torch.float32:
                return torch.from_numpy(self._key.decrypt_f32(tensor._data))
            if tensor.dtype == torch.int64:
                return torch.from_numpy(self._key.decrypt_i64(tensor._data))
            if tensor.dtype == torch.int32:
                return torch.from_numpy(self._key.decrypt_i32(tensor._data))
        elif hasattr(tensor, "decrypt"):
            return tensor.decrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")
