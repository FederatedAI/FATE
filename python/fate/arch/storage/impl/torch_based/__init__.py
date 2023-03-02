from ._storage import _TorchStorage


def encrypt(storage, encryptor):
    return encryptor.encrypt(storage)


_ops_map = {"encrypt": encrypt}
