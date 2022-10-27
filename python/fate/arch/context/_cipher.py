from fate.interface import CipherKit as CipherKitInterface

from ..tensor._phe import PHECipher
from ..unify import device


class CipherKit(CipherKitInterface):
    def __init__(self, device: device) -> None:
        self.device = device

    @property
    def phe(self):
        return PHECipher(self.device)
