"""ops for inside use only"""


def quantile_fi(tensor):
    if isinstance(tensor, torch.Tensor):
        return encryptor.encrypt(tensor)
