from psi_ecdh_curve25519 import ECDHCurve25519
import pickle
import unittest
import random


class TestStringMethods(unittest.TestCase):
    def test_ecdh(self):
        k1 = ECDHCurve25519()
        k2 = ECDHCurve25519()
        m = random.SystemRandom().randbytes(33)
        self.assertEqual(
            k2.diffie_hellman(k1.encrypt(m)), k1.diffie_hellman(k2.encrypt(m))
        )

    def test_pickle(self):
        k1 = ECDHCurve25519()
        m = random.SystemRandom().randbytes(34)
        pickled = pickle.dumps(k1)
        k2 = pickle.loads(pickled)
        self.assertEqual(k1.encrypt(m), k2.encrypt(m))


if __name__ == "__main__":
    unittest.main()
