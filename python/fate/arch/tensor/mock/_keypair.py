import torch

from ._tensor import MockPaillierTensor


class PaillierCipher:
    def __init__(self, **kwargs) -> None:
        ...

    @classmethod
    def keygen(cls, key_length):
        return cls()

    @property
    def pk(self):
        return MockPaillierTensorEncryptor()

    @property
    def coder(self):
        return MockPaillierTensorCoder()

    @property
    def sk(self):
        return MockPaillierTensorDecryptor()


class MockPaillierTensorEncryptor:
    def encrypt(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            return MockPaillierTensor(tensor.detach())
        elif hasattr(tensor, "encrypt"):
            return tensor.encrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def encrypt_encoded(self, tensor: MockPaillierTensor):
        if isinstance(tensor, MockPaillierTensor):
            return MockPaillierTensor(tensor._data)
        elif hasattr(tensor, "encrypt_encoded"):
            return tensor.encrypt_encoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")


class MockPaillierTensorCoder:
    def encode(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach()
        elif hasattr(tensor, "encode"):
            return tensor.encode(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def decode(self, tensor: MockPaillierTensor):
        if isinstance(tensor, torch.Tensor):
            return tensor
        elif hasattr(tensor, "decode"):
            return tensor.decode(self)
        raise NotImplementedError(f"`{tensor}` not supported")


class MockPaillierTensorDecryptor:
    def decrypt(self, tensor: MockPaillierTensor):
        if isinstance(tensor, MockPaillierTensor):
            return tensor._data
        elif hasattr(tensor, "decrypt"):
            return tensor.decrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")

    def decrypt_decoded(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor
        elif hasattr(tensor, "decrypt_decoded"):
            return tensor.decrypt_decoded(self)
        raise NotImplementedError(f"`{tensor}` not supported")
