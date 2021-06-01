import functools
from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import PaillierEncrypt, IterativeAffineEncrypt
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.cipher_compressor.compressor import NormalCipherPackage
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
from federatedml.util import consts
from federatedml.util import LOGGER

precision = 2**53
REGRESSION_MAX_GRADIENT = 10**9


class SplitInfoPackage(NormalCipherPackage):

    def __init__(self, padding_length, max_capacity, round_decimal):
        super(SplitInfoPackage, self).__init__(padding_length, max_capacity, round_decimal)
        self._split_info_without_gh = []
        self._cur_splitinfo_contains = 0

    def add(self, split_info):

        split_info_cp = SplitInfo(sitename=split_info.sitename, best_bid=split_info.best_bid,
                                  best_fid=split_info.best_fid, missing_dir=split_info.missing_dir,
                                  mask_id=split_info.mask_id, sample_count=split_info.sample_count)

        en_g = split_info.sum_grad
        super(SplitInfoPackage, self).add(en_g)
        self._cur_splitinfo_contains += 1
        self._split_info_without_gh.append(split_info_cp)

    def has_space(self):
        return self._capacity_left - 1 >= 0  # g and h

    def unpack(self, decrypter, raw_decrypt=False):
        unpack_rs = super(SplitInfoPackage, self).unpack(decrypter, raw_decrypt=raw_decrypt)
        for split_info, g_h in zip(self._split_info_without_gh, unpack_rs):
            split_info.sum_grad = g_h

        return self._split_info_without_gh


def get_homo_encryption_max_int(encrypter):

    if type(encrypter) == PaillierEncrypt:
        max_pos_int = encrypter.public_key.max_int
        min_neg_int = -max_pos_int
    elif type(encrypter) == IterativeAffineEncrypt:
        n_array = encrypter.key.n_array
        allowed_max_int = n_array[0]
        max_pos_int = int(allowed_max_int * 0.9) - 1  # the other 0.1 part is for negative num
        min_neg_int = (max_pos_int - allowed_max_int) + 1
    else:
        raise ValueError('unknown encryption type')

    return max_pos_int, min_neg_int


class GHPacker(object):

    def __init__(self, pos_max: int, sample_num: int, precision=2**53, max_sample_weight=1.0,
                 task_type=consts.CLASSIFICATION):

        if task_type == consts.CLASSIFICATION:
            g_max = 1.0
            g_min = -1.0
            h_max = 1.0
        elif task_type == consts.REGRESSION:
            g_max = REGRESSION_MAX_GRADIENT  # assign a large value for regression gradients
            g_min = -g_max
            h_max = 2.0
        else:
            raise ValueError('unknown task type {}'.format(task_type))

        self.g_max, self.g_min, self.h_max = g_max * max_sample_weight, g_min * max_sample_weight, h_max * max_sample_weight
        self.g_offset = abs(self.g_min)

        self.g_assign_bit, self.g_modulo, self.g_max_int, self.offset, self.h_modulo, self.h_max_int,\
            self.cipher_compress_capacity = \
            self._bit_assign_suggest(pos_max, sample_num, precision)

        self.total_bit_len = self.g_assign_bit + self.offset
        self.exponent = FixedPointNumber.encode(0, precision=precision).exponent
        self.precision = precision

        LOGGER.debug('total bit len is {}, g offset is {}, g max is {}'.
                     format(self.total_bit_len, self.g_offset, self.g_max))

    def _bit_assign_suggest(self, max_pos: int, sample_num: int, precision=2 ** 53):

        pos_bit_len = max_pos.bit_length()
        h_sum_max = self.h_max * sample_num
        h_max_int = int(h_sum_max * precision)
        h_sum_max_int_bit_len = int(h_sum_max * precision).bit_length() + 1

        g_offset_max = self.g_offset + self.g_max
        g_pos_sum_max_int = int(g_offset_max * sample_num * precision) + 1

        modulo_bit_len = (abs(g_pos_sum_max_int)).bit_length() + 1
        modulo_int = 2 ** modulo_bit_len

        assert modulo_bit_len + h_sum_max_int_bit_len < pos_bit_len, 'no enough bits for packing {} {}'. \
            format(modulo_bit_len + h_sum_max_int_bit_len, pos_bit_len)

        h_assign_bit = h_sum_max_int_bit_len
        h_modulo = 2 ** h_sum_max_int_bit_len
        g_assign_bit = modulo_bit_len
        g_modulo = modulo_int
        g_max_int = g_pos_sum_max_int

        cipher_compress_capacity = pos_bit_len // (h_assign_bit + g_assign_bit)

        return g_assign_bit, g_modulo, g_max_int, h_assign_bit, h_modulo, h_max_int, cipher_compress_capacity

    @staticmethod
    def raw_encrypt(plaintext, encrypter, exponent):

        if type(encrypter) == PaillierEncrypt:
            paillier_num = encrypter.raw_encrypt(plaintext, exponent)
            return paillier_num
        elif type(encrypter) == IterativeAffineEncrypt:
            affine_cipher = encrypter.raw_encrypt(plaintext)
            return affine_cipher
        else:
            raise ValueError('unknown encryption type :{}'.format(type(encrypter)))

    @staticmethod
    def raw_decrypt(cipher, encrypter):

        decrypt_rs = encrypter.raw_decrypt(cipher)
        return decrypt_rs

    @staticmethod
    def encode(num, mul, modulo):
        int_fixpoint = int(round(num * mul))
        return int_fixpoint % modulo

    @staticmethod
    def pack_func(gh, mul, g_modulo, h_modulo, offset, g_offset, ret_fixed_point=False, exponent=0, n=0):

        g, h = gh[0], gh[1]
        if abs(g) >= REGRESSION_MAX_GRADIENT:
            g = REGRESSION_MAX_GRADIENT
        g += g_offset  # to positive
        g_encoding = GHPacker.encode(g, mul, g_modulo)
        h_encoding = GHPacker.encode(h, mul, h_modulo)
        pack_num = (g_encoding << offset) + h_encoding
        if ret_fixed_point:
            pack_num = FixedPointNumber(encoding=pack_num, exponent=exponent, n=n)
        return pack_num

    @staticmethod
    def unpack_func(g_h_plain_text, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent):

        g = g_h_plain_text >> h_assign_bit
        g = g % g_modulo
        g_fix_num = FixedPointNumber(g, exponent, g_modulo, g_max_int)
        h_valid_mask = (1 << h_assign_bit) - 1
        h = g_h_plain_text & h_valid_mask
        h = h % h_modulo
        h_fix_num = FixedPointNumber(h, exponent, h_modulo, h_max_int)
        return g_fix_num.decode(), h_fix_num.decode()

    def pack(self, gh):

        exponent = FixedPointNumber.encode(0, self.g_modulo, self.g_max_int, precision).exponent
        mul = pow(FixedPointNumber.BASE, exponent)
        pack_num = self.pack_func(gh, mul, self.g_modulo, self.h_modulo, self.offset, self.g_offset)
        return pack_num

    def unpack(self, en_num, encrypter, offset_sample_num, remove_offset=True):

        de_rs = self.raw_decrypt(en_num, encrypter)
        g, h = self.unpack_func(de_rs, self.offset, self.g_modulo, self.g_max_int, self.h_modulo, self.h_max_int,
                                self.exponent)
        if remove_offset:
            g = g - offset_sample_num * self.g_offset
        return g, h

    def pack_and_encrypt(self, gh, encrypter):

        exponent = FixedPointNumber.encode(0, self.g_modulo, self.g_max_int, precision).exponent
        mul = pow(FixedPointNumber.BASE, exponent)
        pack_num = self.pack_func(gh, mul, self.g_modulo, self.h_modulo, self.offset, self.g_offset)
        encrypt_num = self.raw_encrypt(pack_num, encrypter, exponent=exponent)
        return encrypt_num, 0

    def pack_and_encrypt_2(self, gh_table, encrypt_calculator):

        exponent = FixedPointNumber.encode(0, self.g_modulo, self.g_max_int, precision).exponent
        mul = pow(FixedPointNumber.BASE, exponent)
        ret_fixed_point = False
        paillier_n = 0
        if encrypt_calculator.mode != 'strict':
            if type(encrypt_calculator.encrypter) == PaillierEncrypt:
                ret_fixed_point = True
                paillier_n = encrypt_calculator.encrypter.public_key.n
            else:
                raise ValueError("non paillier encryption does not support 'fast' or 'strict' mode")

        pack_func = functools.partial(self.pack_func, mul=mul, g_modulo=self.g_modulo, h_modulo=self.h_modulo,
                                      offset=self.offset, g_offset=self.g_offset,
                                      ret_fixed_point=ret_fixed_point,
                                      exponent=exponent, n=paillier_n)
        pack_gh_table = gh_table.mapValues(pack_func)
        rs_table = encrypt_calculator.raw_encrypt(pack_gh_table, exponent=exponent)
        rs_table = rs_table.mapValues(lambda x: (x, 0))
        return rs_table

    def decompress_and_unpack(self, split_info_package_list, encrypter):

        decompressor = PackedGHDecompressor(encrypter)
        split_info_list = decompressor.unpack_split_info(split_info_package_list)
        for split_info in split_info_list:
            g, h = self.unpack_func(split_info.sum_grad, self.offset, self.g_modulo, self.g_max_int, self.h_modulo,
                                    self.h_max_int, self.exponent)
            split_info.sum_grad = g - split_info.sample_count * self.g_offset
            split_info.sum_hess = h
        return split_info_list


class PackedGHCompressor(object):

    def __init__(self, padding_bit_len, max_capacity):
        self.padding_bit_len = padding_bit_len
        self.max_capacity = max_capacity

    def compress_split_info(self, split_info_list, g_h_sum_info):

        split_info_list.append(g_h_sum_info)  # append to end
        rs = []
        cur_package = SplitInfoPackage(self.padding_bit_len, self.max_capacity, 0)
        for s in split_info_list:
            if not cur_package.has_space():
                rs.append(cur_package)
                cur_package = SplitInfoPackage(self.padding_bit_len, self.max_capacity, 0)
            cur_package.add(s)
        rs.append(cur_package)
        return rs


class PackedGHDecompressor(object):

    def __init__(self, encrypter):
        self.encrypter = encrypter

    def unpack_split_info(self, packages):
        rs_list = []
        for p in packages:
            rs_list.extend(p.unpack(self.encrypter, raw_decrypt=True))
        return rs_list
