import unittest

import numpy as np
from federatedml.cipher_compressor.compressor import PackingCipherTensor
from federatedml.secureprotol import PaillierEncrypt


class TestXgboostCriterion(unittest.TestCase):

    def setUp(self):
        print('init testing')

    def test_plain_add_sub_mul(self):

        a = PackingCipherTensor([1, 2, 3, 4])
        b = PackingCipherTensor([2, 3, 4, 5])
        c = PackingCipherTensor(124)
        d = PackingCipherTensor([114514])

        print(a + b)
        print(b + a)
        print(c * 123)
        print(d * 314)
        print(12 * a)
        print(a * 2)
        print(a / 12)
        print(b - a)
        print(d + 3)
        print('plain test done')
        print('*' * 30)

    def test_cipher_add_sub_mul(self):

        encrypter = PaillierEncrypt()
        encrypter.generate_key(1024)
        en_1, en_2, en_3, en_4 = encrypter.encrypt(1), encrypter.encrypt(2), encrypter.encrypt(3), encrypter.encrypt(4)
        en_5, en_6, en_7, en_8 = encrypter.encrypt(5), encrypter.encrypt(6), encrypter.encrypt(7), encrypter.encrypt(8)
        a = PackingCipherTensor([en_1, en_2, en_3, en_4])
        b = PackingCipherTensor([en_5, en_6, en_7, en_8])
        c = PackingCipherTensor(encrypter.encrypt(1))
        d = PackingCipherTensor([encrypter.encrypt(5)])

        rs_1 = a + b
        rs_2 = b - a
        rs_3 = c + d
        rs_4 = 123 * c
        rs_5 = d * 456
        rs_6 = a * 114
        print(encrypter.recursive_decrypt(rs_1.ciphers))
        print(encrypter.recursive_decrypt(rs_2.ciphers))
        print(encrypter.recursive_decrypt(rs_3.ciphers))
        print(encrypter.decrypt(rs_4.ciphers))
        print(encrypter.decrypt(rs_5.ciphers))
        print(encrypter.recursive_decrypt(rs_6.ciphers))
        print('cipher test done')
        print('*' * 30)


if __name__ == '__main__':
    unittest.main()
