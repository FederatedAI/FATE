import fate_tensor
import pickle
import unittest
import numpy as np


class TestBaseMethods(unittest.TestCase):
    def setUp(self):
        self.a = np.random.random((10, 10))
        self.b = np.random.random((10, 10))
        self.pk, self.ek = fate_tensor.keygen(1024)

    def test_keygen(self):
        fate_tensor.keygen(1024)

    def test_cipher(self):
        self.assertTrue(
            np.isclose(self.ek.decrypt_f64(self.pk.encrypt_f64(self.a)), self.a).all()
        )

    def test_add(self):
        ea = self.pk.encrypt_f64(self.a)
        eb = self.pk.encrypt_f64(self.b)
        e = self.ek.decrypt_f64(ea.add_cipherblock(eb)) - (self.a + self.b)
        self.assertTrue(np.isclose(e, 0).all())

    def test_sub(self):
        ea = self.pk.encrypt_f64(self.a)
        eb = self.pk.encrypt_f64(self.b)
        e = self.ek.decrypt_f64(ea.sub_cipherblock(eb)) - (self.a - self.b)
        self.assertTrue(np.isclose(e, 0).all())

    def test_mul(self):
        ea = self.pk.encrypt_f64(self.a)
        e = self.ek.decrypt_f64(ea.mul_plaintext(self.b)) - (self.a * self.b)
        self.assertTrue(np.isclose(e, 0).all())

    def test_par_add(self):
        ea = self.pk.encrypt_f64_par(self.a)
        eb = self.pk.encrypt_f64_par(self.b)
        e = self.ek.decrypt_f64_par(ea.add_cipherblock_par(eb)) - (self.a + self.b)
        self.assertTrue(np.isclose(e, 0).all())


if __name__ == "__main__":
    unittest.main()
