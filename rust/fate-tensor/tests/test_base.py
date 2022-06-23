import fate_tensor
import pickle
import unittest
import numpy as np


class TestBaseMethods(unittest.TestCase):
    def setUp(self):
        self.af64 = np.random.random((10, 10)).astype(np.float64)
        self.bf64 = np.random.random((10, 10)).astype(np.float64)
        self.af32 = np.random.random((10, 10)).astype(np.float32)
        self.bf32 = np.random.random((10, 10)).astype(np.float32)
        self.pk, self.sk = fate_tensor.keygen(1024)

    def encrypt(self, fp, par, data):
        return getattr(self.pk, f"encrypt_{fp}{par}")(data)

    def decrypt(self, fp, par, data):
        return getattr(self.sk, f"decrypt_{fp}{par}")(data)

    def data(self, fp, index):
        return getattr(self, "{}{}".format(["a", "b"][index], fp))

    def test_keygen(self):
        fate_tensor.keygen(1024)

    def test_cipher(self):
        for par in ["", "_par"]:
            for fp in ["f64", "f32"]:
                diff = self.decrypt(fp, par, self.encrypt(fp, par, self.data(fp, 0))) - self.data(fp, 0)
                self.assertTrue(np.isclose(diff, 0).all())

    def test_add_cipher(self):
        for par in ["", "_par"]:
            for fp in ["f64", "f32"]:
                ea = self.encrypt(fp, par, self.data(fp, 0))
                eb = self.encrypt(fp, par, self.data(fp, 1))
                ec = getattr(ea, f"add_cipherblock{par}")(eb)
                c = self.data(fp, 0) + self.data(fp, 1)
                diff = self.decrypt(fp, par, ec) - c
                self.assertTrue(np.isclose(diff, 0).all())

    def test_add_plaintext(self):
        for par in ["", "_par"]:
            for fp in ["f64", "f32"]:
                ea = self.encrypt(fp, par, self.data(fp, 0))
                b = self.data(fp, 1)
                ec = getattr(ea, f"add_plaintext_{fp}{par}")(b)
                c = self.data(fp, 0) + self.data(fp, 1)
                diff = self.decrypt(fp, par, ec) - c
                self.assertTrue(np.isclose(diff, 0).all())

    def test_sub_cipher(self):
        for par in ["", "_par"]:
            for fp in ["f64", "f32"]:
                ea = self.encrypt(fp, par, self.data(fp, 0))
                eb = self.encrypt(fp, par, self.data(fp, 1))
                ec = getattr(ea, f"sub_cipherblock{par}")(eb)
                c = self.data(fp, 0) - self.data(fp, 1)
                diff = self.decrypt(fp, par, ec) - c
                self.assertTrue(np.isclose(diff, 0).all())

    def test_sub_plaintext(self):
        for par in ["", "_par"]:
            for fp in ["f64", "f32"]:
                ea = self.encrypt(fp, par, self.data(fp, 0))
                b = self.data(fp, 1)
                ec = getattr(ea, f"sub_plaintext_{fp}{par}")(b)
                c = self.data(fp, 0) - self.data(fp, 1)
                diff = self.decrypt(fp, par, ec) - c
                self.assertTrue(np.isclose(diff, 0).all())

    def test_mul(self):
        for par in ["", "_par"]:
            for fp in ["f64", "f32"]:
                ea = self.encrypt(fp, par, self.data(fp, 0))
                b = self.data(fp, 1)
                ec = getattr(ea, f"mul_plaintext_{fp}{par}")(b)
                c = self.data(fp, 0) * self.data(fp, 1)
                diff = self.decrypt(fp, par, ec) - c
                self.assertTrue(np.isclose(diff, 0).all())


if __name__ == "__main__":
    unittest.main()
