from typing import Protocol


class Cipher(Protocol):
    def keygen(self):
        ...
