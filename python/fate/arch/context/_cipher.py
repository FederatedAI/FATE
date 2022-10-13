from fate.interface import CipherKit as CipherKitInterface

from ..tensor._phe import PHECipher
from ..unify import Backend, device


class CipherKit(CipherKitInterface):
    def __init__(self, backend: Backend, device: device) -> None:
        self.backend = backend
        self.device = device

    @property
    def phe(self):
        return PHECipher(self.backend, self.device)
