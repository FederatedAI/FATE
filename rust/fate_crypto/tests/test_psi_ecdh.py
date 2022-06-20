from fate_crypto.psi import Curve25519
import pickle
import unittest
import random


class TestStringMethods(unittest.TestCase):
    def test_ecdh(self):
        k1 = Curve25519()
        k2 = Curve25519()
        m = random.SystemRandom().getrandbits(33 * 8).to_bytes(33, "little")
        self.assertEqual(
            k2.diffie_hellman(k1.encrypt(m)), k1.diffie_hellman(k2.encrypt(m))
        )

    def test_pickle(self):
        k1 = Curve25519()
        m = random.SystemRandom().getrandbits(33 * 8).to_bytes(33, "little")
        pickled = pickle.dumps(k1)
        k2 = pickle.loads(pickled)
        self.assertEqual(k1.encrypt(m), k2.encrypt(m))


if __name__ == "__main__":
    unittest.main()
