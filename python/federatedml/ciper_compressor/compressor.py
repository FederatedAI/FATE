import math
from abc import ABC
from abc import abstractmethod
from federatedml.util import consts
from typing import List


class CipherPackage(ABC):

    @abstractmethod
    def add(self, obj):
        pass

    @abstractmethod
    def unpack(self, decrypter):
        pass

    @abstractmethod
    def has_space(self):
        pass


class NormalCipherPackage(CipherPackage):

    def __init__(self, padding_length, max_capacity, round_decimal=7):

        self._round_decimal = round_decimal
        self._padding_num = 2 ** padding_length
        self.max_capacity = max_capacity
        self._cipher_text = None
        self._capacity_left = max_capacity
        self._has_space = True

    def add(self, cipher_text):

        if self._capacity_left == 0:
            raise ValueError('cipher number exceeds package max capacity')

        if self._cipher_text is None:
            self._cipher_text = cipher_text
        else:
            self._cipher_text = self._cipher_text * self._padding_num
            self._cipher_text = self._cipher_text + cipher_text

        self._capacity_left -= 1
        if self._capacity_left == 0:
            self._has_space = False

    def unpack(self, decrypter):

        unpack_result = []
        compressed_plain_text = int(decrypter.decrypt(self._cipher_text))
        bit_len = (self._padding_num - 1).bit_length()
        for i in range(self.cur_cipher_contained()):
            num = (compressed_plain_text & (self._padding_num - 1)) / (10 ** self._round_decimal)
            compressed_plain_text = compressed_plain_text >> bit_len
            unpack_result.insert(0, num)

        return unpack_result

    def has_space(self):
        return self._has_space

    def cur_cipher_contained(self):
        return self.max_capacity - self._capacity_left

    def retrieve(self):
        return self._cipher_text


class CipherEncoder(object):  # this class encode to large integer

    def __init__(self, round_decimal):
        self.round_decimal = round_decimal

    def encode(self, num):
        return int(num * 10**self.round_decimal)

    def encode_list(self, plaintext_list):

        int_list = []
        for i in plaintext_list:
            int_list.append(self.encode(i))
        return int_list

    def encode_and_encrypt(self, plaintext_list, encrypter):
        int_list = self.encode_list(plaintext_list)
        return [encrypter.encrypt(i) for i in int_list]


class CipherDecompressor(object):  # this class endcode and unzip cipher package

    def __init__(self, encrypter):
        self.encrypter = encrypter

    def unpack(self, packages: List[CipherPackage]):

        rs_list = []
        for p in packages:
            rs_list.append(p.unpack(self.encrypter))

        return rs_list


class CipherCompressor(object):

    def __init__(self, cipher_type, max_float, max_capacity_int, package_class, round_decimal):

        """
        Parameters
        ----------
        cipher_type: paillier only
        max_float： the max number of ciphertext
        max_capacity_int: the max number allowed of current encrypt algorithm
        package_class: cipher package type, can be customized, need implement "add" and "unpack"
        round_decimal: decimal rounding setting
        """

        if cipher_type != consts.PAILLIER and cipher_type != consts.ITERATIVEAFFINE:
            raise ValueError('encrypt type {} is not supported by cipher compressing'.format(cipher_type))

        self._ciper_type = cipher_type
        self.max_float = max_float
        self.max_capacity_int = max_capacity_int
        self._package_class = package_class
        self.round_decimal = round_decimal
        self._padding_length, self.max_capacity = self.advise(max_float, max_capacity_int, cipher_type, round_decimal)

    @staticmethod
    def advise(max_float, max_capacity_int, cipher_type=consts.PAILLIER, round_decimal=7):

        max_int = int(max_float * (10**round_decimal))
        key_length = max_capacity_int.bit_length()
        padding_length = max_int.bit_length()

        if cipher_type == consts.PAILLIER:
            cipher_capacity = (key_length - 1) // padding_length
        else:
            raise ValueError('Non paillier method is not supported')

        if cipher_capacity <= 1:
            raise ValueError('cipher package capacity is too small! capacity is: {}.'
                             'compressing parameters are: max_float {}, round_decmial {},'
                             'key_length {}'.format(cipher_capacity, max_float, round_decimal, key_length))

        return padding_length, cipher_capacity

    def compress(self, cipher_text_list):

        rs = []
        cur_package = self._package_class(self._padding_length, self.max_capacity, self.round_decimal)
        for c in cipher_text_list:
            if not cur_package.has_space():
                rs.append(cur_package)
                cur_package = self._package_class(self._padding_length, self.max_capacity, self.round_decimal)
            cur_package.add(c)

        rs.append(cur_package)
        return rs


if __name__ == '__main__':
    import numpy as np
    from federatedml.secureprotol import PaillierEncrypt as Encrypt

    int_num = 1000
    decimal_to_keep = 7
    key_length = 1024
    test_nums = np.random.random(2)
    test_nums += int_num

    en = Encrypt()
    en.generate_key(key_length)
    max_float = test_nums.max()
    cipher_max_int = en.public_key.max_int

    encoder = CipherEncoder(round_decimal=7)
    compressor = CipherCompressor(consts.PAILLIER, max_float, cipher_max_int, NormalCipherPackage, decimal_to_keep)
    decompressor = CipherDecompressor(encrypter=en)

    en_list = encoder.encode_and_encrypt(test_nums, encrypter=en)
    packages = compressor.compress(en_list)
    rs = decompressor.unpack(packages)

    print(test_nums)
    print(np.array(rs))
