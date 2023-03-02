from fate.arch.unify import device


def paillier_keygen(_device, key_length):
    if _device == device.CPU:
        from .impl.cpu_paillier.cpu_paillier_block import BlockPaillierCipher

        encryptor, decryptor = BlockPaillierCipher().keygen(key_length=key_length)
        return EncryptorTorchToRustPaillier(encryptor), DecryptorTorchToRustPaillier(decryptor)
    raise NotImplementedError()


class PHECipher:
    @classmethod
    def keygen(cls, **kwargs):
        from .impl.cpu_paillier.cpu_paillier_block import BlockPaillierCipher

        key_length = kwargs.get("key_length", 1024)
        encryptor, decryptor = BlockPaillierCipher().keygen(key_length=key_length)
        return EncryptorTorchToRustPaillier(encryptor), DecryptorTorchToRustPaillier(decryptor)


class EncryptorTorchToRustPaillier:
    def __init__(self, inside) -> None:
        self.inside = inside

    def encrypt(self, storage):
        from ._dtype import dtype
        from .impl.cpu_paillier.paillier import _RustPaillierStorage
        from .impl.torch_based._storage import _TorchStorage

        assert isinstance(storage, _TorchStorage)
        return _RustPaillierStorage(dtype.paillier, storage.shape, self.inside.encrypt(storage.data))


class DecryptorTorchToRustPaillier:
    def __init__(self, inside) -> None:
        self.inside = inside

    def decrypt(self, storage):
        from ._dtype import dtype
        from .impl.cpu_paillier.paillier import _RustPaillierStorage
        from .impl.torch_based._storage import _TorchStorage

        assert isinstance(storage, _RustPaillierStorage)
        return _TorchStorage(dtype.paillier, storage.shape, self.inside.decrypt(storage.data))
