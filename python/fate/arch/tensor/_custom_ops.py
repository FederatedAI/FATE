import torch


def encrypt(tensor, encryptor):
    if isinstance(tensor, torch.Tensor):
        return encryptor.encrypt(tensor)
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(encrypt, (type(tensor),), (tensor, encryptor), None)
    raise NotImplementedError("")


def decrypt(tensor, decryptor):
    if isinstance(tensor, torch.Tensor):
        return decryptor.encrypt(tensor)
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(decrypt, (type(tensor),), (tensor, decryptor), None)
    raise NotImplementedError("")


def rmatmul(input, other):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        return torch.matmul(other, input)
    else:
        # torch tensor-like
        if isinstance(input, torch.Tensor):
            return torch.matmul(other, input)

        else:
            if hasattr(input, "__torch_function__"):
                return input.__torch_function__(rmatmul, (type(input), type(other)), (input, other), None)
    raise NotImplementedError("")


def to_local(input):
    if isinstance(input, torch.Tensor):
        return input

    else:
        # torch tensor-like
        if hasattr(input, "__torch_function__"):
            return input.__torch_function__(to_local, (type(input),), (input,), None)
    raise NotImplementedError("")


# hook custom ops to torch
torch.encrypt = encrypt
torch.decrypt = decrypt
torch.rmatmul = rmatmul
torch.to_local = to_local
