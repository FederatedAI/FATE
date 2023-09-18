from fate_utils.psi import Curve25519
import pickle
import unittest
import random


class TestStringMethods(unittest.TestCase):
    def test_ecdh(self):
        k1 = Curve25519()
        k2 = Curve25519()
        m = random.SystemRandom().getrandbits(33 * 8).to_bytes(33, "little")
        self.assertEqual(k2.diffie_hellman(k1.encrypt(m)), k1.diffie_hellman(k2.encrypt(m)))

    def test_ecdh_vec(self):
        k1 = Curve25519()
        k2 = Curve25519()
        m = [random.SystemRandom().getrandbits(33 * 8).to_bytes(33, "little") for _ in range(100)]
        s1 = k1.encrypt_vec(m)
        s12 = k2.diffie_hellman_vec(s1)
        s2 = k2.encrypt_vec(m)
        s21 = k1.diffie_hellman_vec(s2)
        self.assertEqual(s12, s21)

    def test_pickle(self):
        k1 = Curve25519()
        m = random.SystemRandom().getrandbits(33 * 8).to_bytes(33, "little")
        pickled = pickle.dumps(k1)
        k2 = pickle.loads(pickled)
        self.assertEqual(k1.encrypt(m), k2.encrypt(m))


if __name__ == "__main__":
    unittest.main()
