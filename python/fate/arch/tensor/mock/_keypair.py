import torch

from ._tensor import MockPaillierTensor


def keygen(key_length):
    return MockPaillierTensorEncryptor(), MockPaillierTensorDecryptor()


class MockPaillierTensorEncryptor:
    def encrypt(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            return MockPaillierTensor(tensor.detach().numpy(), tensor.dtype)
        elif hasattr(tensor, "encrypt"):
            return tensor.encrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")


class MockPaillierTensorDecryptor:
    def decrypt(self, tensor: MockPaillierTensor):
        if isinstance(tensor, MockPaillierTensor):
            return torch.from_numpy(tensor._data)
        elif hasattr(tensor, "decrypt"):
            return tensor.decrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")
