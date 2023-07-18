from .mock import keygen as mock_keygen
from .paillier import keygen as paillier_keygen


def phe_keygen(kind, options):
    if kind == "paillier":
        return paillier_keygen(**options)
    elif kind == "mock":
        return mock_keygen(**options)
    else:
        raise ValueError(f"Unknown PHE keygen kind: {kind}")


class PHEKit:
    def __init__(self, kind, key_length) -> None:
        if kind == "paillier":
            self._encryptor, self._decryptor = paillier_keygen(key_length)
        elif kind == "mock":
            self._encryptor, self._decryptor = mock_keygen(key_length)
        else:
            raise ValueError(f"Unknown PHE keygen kind: {kind}")

    @property
    def encryptor(self):
        return self._encryptor

    @property
    def decryptor(self):
        return self._decryptor
