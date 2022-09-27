from typing import Protocol


class PHECipher(Protocol):
    def keygen(self, kind, options={}):
        ...


class CipherKit(Protocol):
    phe: PHECipher
