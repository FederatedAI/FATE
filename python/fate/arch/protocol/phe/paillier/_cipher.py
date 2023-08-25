import torch
from fate_utils.paillier import CiphertextVector, PlaintextVector, Ciphertext, Plaintext
from fate_utils.paillier import PK as _PK
from fate_utils.paillier import SK as _SK
from fate_utils.paillier import keygen as _keygen

V = torch.Tensor
E = Ciphertext
F = Plaintext
EV = CiphertextVector
FV = PlaintextVector


class PK:
    def __init__(self, pk: _PK):
        self.pk = pk

    def encrypt_encoded(self, vec: FV, obfuscate: bool) -> EV:
        from ._evaluator import Evaluator
        return Evaluator.encrypt(self, vec, obfuscate)

    def encrypt_encoded_scalar(self, val, obfuscate) -> E:
        from ._evaluator import Evaluator
        return Evaluator.encrypt_scalar(self, val, obfuscate)


class SK:
    def __init__(self, sk: _SK):
        self.sk = sk

    def decrypt_to_encoded(self, vec: EV) -> FV:
        from ._evaluator import Evaluator
        return Evaluator.decrypt(self, vec)


def keygen(key_size):
    sk, pk = _keygen(key_size)
    return SK(sk), PK(pk)
