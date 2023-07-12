import fate_utils
import torch

from ._tensor import PaillierTensor


def keygen(key_length):
    encryptor, decryptor = fate_utils.tensor.keygen(key_length)
    return PaillierTensorEncryptor(encryptor), PaillierTensorDecryptor(decryptor)


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


class PaillierTensorDecryptor:
    def __init__(self, key) -> None:
        self._key = key

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
