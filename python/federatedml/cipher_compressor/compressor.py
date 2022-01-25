import math
from abc import ABC
from abc import abstractmethod
from federatedml.util import LOGGER
from federatedml.secureprotol import PaillierEncrypt
from federatedml.transfer_variable.transfer_class.cipher_compressor_transfer_variable \
    import CipherCompressorTransferVariable


def get_homo_encryption_max_int(encrypter):

    if isinstance(encrypter, PaillierEncrypt):
        max_pos_int = encrypter.public_key.max_int
        min_neg_int = -max_pos_int
    else:
        raise ValueError('unknown encryption type')

    return max_pos_int, min_neg_int


def cipher_compress_advisor(encrypter, plaintext_bit_len):

    max_pos_int, min_neg_int = get_homo_encryption_max_int(encrypter)
    max_bit_len = max_pos_int.bit_length()
    capacity = max_bit_len // plaintext_bit_len
    return capacity


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


class PackingCipherTensor(object):

    """
    A naive realization of cipher tensor
    """

    def __init__(self, ciphers):

        if isinstance(ciphers, list):
            if len(ciphers) == 1:
                self.ciphers = ciphers[0]
            else:
                self.ciphers = ciphers
            self.dim = len(ciphers)
        else:
            self.ciphers = ciphers
            self.dim = 1

    def __add__(self, other):

        new_cipher_list = []
        if isinstance(other, PackingCipherTensor):
            assert self.dim == other.dim

            if self.dim == 1:
                return PackingCipherTensor(self.ciphers + other.ciphers)
            for c1, c2 in zip(self.ciphers, other.ciphers):
                new_cipher_list.append(c1 + c2)
            return PackingCipherTensor(ciphers=new_cipher_list)
        else:
            # scalar / single en num
            if self.dim == 1:
                return PackingCipherTensor(self.ciphers + other)
            for c in self.ciphers:
                new_cipher_list.append(c + other)
            return PackingCipherTensor(ciphers=new_cipher_list)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return other + (self * -1)

    def __mul__(self, other):

        if self.dim == 1:
            return PackingCipherTensor(self.ciphers * other)
        new_cipher_list = []
        for c in self.ciphers:
            new_cipher_list.append(c * other)
        return PackingCipherTensor(new_cipher_list)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __repr__(self):
        return "[" + self.ciphers.__repr__() + "], dim {}".format(self.dim)


class NormalCipherPackage(CipherPackage):

    def __init__(self, padding_length, max_capacity):

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

        if isinstance(decrypter, PaillierEncrypt):
            compressed_plain_text = decrypter.privacy_key.raw_decrypt(self._cipher_text.ciphertext())
        else:
            raise ValueError('unknown decrypter: {}'.format(type(decrypter)))

        if self.cur_cipher_contained() == 1:
            return [compressed_plain_text]

        unpack_result = []
        bit_len = (self._padding_num - 1).bit_length()
        for i in range(self.cur_cipher_contained()):
            num = (compressed_plain_text & (self._padding_num - 1))
            compressed_plain_text = compressed_plain_text >> bit_len
            unpack_result.insert(0, num)

        return unpack_result

    def has_space(self):
        return self._has_space

    def cur_cipher_contained(self):
        return self.max_capacity - self._capacity_left

    def retrieve(self):
        return self._cipher_text


class PackingCipherTensorPackage(CipherPackage):

    """
    A naive realization of compressible tensor(only compress last dimension because previous ciphers have
    no space for compressing)
    """

    def __init__(self, padding_length, max_capcity):
        self.cached_list = []
        self.compressed_cipher = []
        self.compressed_dim = -1
        self.not_compress_len = None
        self.normal_package = NormalCipherPackage(padding_length, max_capcity)

    def add(self, obj: PackingCipherTensor):

        if self.normal_package.has_space():
            if obj.dim == 1:
                self.normal_package.add(obj.ciphers)
            else:
                self.cached_list.extend(obj.ciphers[:-1])
                self.not_compress_len = len(obj.ciphers[:-1])
                self.normal_package.add(obj.ciphers[-1])
        else:
            raise ValueError('have no space for compressing')

    def unpack(self, decrypter):

        compressed_part = self.normal_package.unpack(decrypter)
        de_rs = []
        if len(self.cached_list) != 0:
            de_rs = decrypter.recursive_raw_decrypt(self.cached_list)

        if len(de_rs) == 0:
            return [[i] for i in compressed_part]
        else:
            rs = []
            idx_0, idx_1 = 0, 0
            while idx_0 < len(self.cached_list):
                rs.append(de_rs[idx_0: idx_0 + self.not_compress_len] + [compressed_part[idx_1]])
                idx_0 += self.not_compress_len
                idx_1 += 1
            return rs

    def has_space(self):
        return self.normal_package.has_space()


class CipherCompressorHost(object):

    def __init__(self, package_class=PackingCipherTensorPackage, sync_para=True):
        """
        Parameters
        ----------
        package_class type of compressed packages
        """

        self._package_class = package_class
        self._padding_length, self._capacity = None, None
        if sync_para:
            self.transfer_var = CipherCompressorTransferVariable()
            # received from host
            self._padding_length, self._capacity = self.transfer_var.compress_para.get(idx=0)
            LOGGER.debug("received parameter from guest is {} {}".format(self._padding_length, self._capacity))

    def compress(self, encrypted_obj_list):

        rs = []
        encrypted_obj_list = list(encrypted_obj_list)
        cur_package = self._package_class(self._padding_length, self._capacity)
        for c in encrypted_obj_list:
            if not cur_package.has_space():
                rs.append(cur_package)
                cur_package = self._package_class(self._padding_length, self._capacity)
            cur_package.add(c)

        rs.append(cur_package)
        return rs

    def compress_dtable(self, table):
        rs = table.mapValues(self.compress)
        return rs


if __name__ == '__main__':
    a = PackingCipherTensor([1, 2, 3, 4])
    b = PackingCipherTensor([2, 3, 4, 5])
    c = PackingCipherTensor(124)
    d = PackingCipherTensor([114514])
