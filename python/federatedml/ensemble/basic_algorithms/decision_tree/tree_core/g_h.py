from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import PaillierEncrypt, IterativeAffineEncrypt
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitinfo_cipher_compressor import SplitInfoPackage
from federatedml.util import LOGGER

VERY_SMALL_FLOAT = -0.000000000001  # for computing edge cases
precision = 2**53

RESERVED_BIT = 1

# range of gradient and hessian
G_MAX = 1.0
G_MIN = -1.0
H_MAX = 1.0

G_OFFSET = 1.0


def iter_fix_point_test(float_num, precision):
    fix_point_int = int(float_num * precision)
    return fix_point_int


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


def bit_assign_suggest(max_pos: int, min_neg: int, sample_num: int, precision=2**53, tree_depth=None):

    pos_bit_len = max_pos.bit_length()
    h_sum_max = H_MAX * sample_num
    h_max_int = int(h_sum_max * precision)
    h_sum_max_int_bit_len = int(h_sum_max * precision).bit_length() + 1

    g_offset_max = G_OFFSET + G_MAX
    g_pos_sum_max_int = int(g_offset_max * sample_num * precision) + 1

    modulo_bit_len = (abs(g_pos_sum_max_int)).bit_length() + 1
    modulo_int = 2 ** modulo_bit_len

    assert modulo_bit_len + h_sum_max_int_bit_len < pos_bit_len, 'no enough bits for packing {} {}'.\
        format(modulo_bit_len+h_sum_max_int_bit_len, pos_bit_len)

    h_assign_bit = h_sum_max_int_bit_len
    h_modulo = 2**h_sum_max_int_bit_len
    g_assign_bit = modulo_bit_len
    g_modulo = modulo_int
    g_max_int = g_pos_sum_max_int

    cipher_compress_capacity = pos_bit_len // (h_assign_bit + g_assign_bit)

    return g_assign_bit, g_modulo, g_max_int, h_assign_bit, h_modulo, h_max_int, cipher_compress_capacity


def encode(num, mul, modulo):
    int_fixpoint = int(round(num * mul))
    return int_fixpoint % modulo


def ghpack(gh, mul, g_modulo, h_modulo, offset):

    g, h = gh[0], gh[1]
    g += G_OFFSET  # become positive
    g_encoding = encode(g, mul, g_modulo)
    h_encoding = encode(h, mul, h_modulo)
    pack_num = (g_encoding << offset) + h_encoding
    return pack_num


def pack_and_encrypt(gh, g_modulo, g_max_int, h_modulo, h_max_int, offset, encrypter, precision=2**53):

    exponent = FixedPointNumber.encode(0, g_modulo, g_max_int, precision).exponent
    mul = pow(FixedPointNumber.BASE, exponent)
    pack_num = ghpack(gh, mul, g_modulo, h_modulo, offset)
    encrypt_num = raw_encrypt(pack_num, encrypter, exponent=exponent)
    return encrypt_num, 0


def unpack(g_h_plain_text, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent):

    g = g_h_plain_text >> h_assign_bit
    g = g % g_modulo
    g_fix_num = FixedPointNumber(g, exponent, g_modulo, g_max_int)
    h_valid_mask = (1 << h_assign_bit) - 1
    h = g_h_plain_text & h_valid_mask
    h = h % h_modulo
    h_fix_num = FixedPointNumber(h, exponent, h_modulo, h_max_int)
    return g_fix_num.decode(), h_fix_num.decode()


def raw_encrypt(plaintext, encrypter, exponent):

    if type(encrypter) == PaillierEncrypt:
        ciphertext = encrypter.public_key.raw_encrypt(plaintext)
        paillier_num = PaillierEncryptedNumber(encrypter.public_key, ciphertext, exponent)
        return paillier_num
    elif type(encrypter) == IterativeAffineEncrypt:
        affine_cipher = encrypter.key.raw_encrypt(plaintext)
        return affine_cipher
    else:
        raise ValueError('unknown encryption type :{}'.format(type(encrypter)))


def raw_decrypt(cipher, encrypter):

    if type(encrypter) == PaillierEncrypt:
        decrypt_rs = encrypter.privacy_key.raw_decrypt(cipher.ciphertext())
        return decrypt_rs
    elif type(encrypter) == IterativeAffineEncrypt:
        decrypt_rs = encrypter.key.raw_decrypt(cipher)
    else:
        raise ValueError('unknown encryption type :{}'.format(type(encrypter)))

    return decrypt_rs


class GHPacker(object):

    def __init__(self, pos_max: int, neg_min:int, sample_num: int, precision=2**53):

        self.g_assign_bit, self.g_modulo, self.g_max_int, self.offset, self.h_modulo, self.h_max_int,\
            self.cipher_compress_capacity = \
            bit_assign_suggest(pos_max, neg_min, sample_num, precision)

        self.total_bit_len = self.g_assign_bit + self.offset
        self.exponent = FixedPointNumber.encode(0, precision=precision).exponent
        self.precision = precision

    def pack(self, gh, encrypter):

        en_num = pack_and_encrypt(gh, self.g_modulo, self.g_max_int, self.h_modulo, self.h_max_int, self.offset,
                                  encrypter, self.precision)
        return en_num

    def unpack(self, en_num, encrypter, offset_sample_num, remove_offset=True):

        de_rs = raw_decrypt(en_num, encrypter)
        g, h = unpack(de_rs, self.offset, self.g_modulo, self.g_max_int, self.h_modulo, self.h_max_int,
                      self.exponent)
        if remove_offset:
            g = g - offset_sample_num * G_OFFSET
        return g, h

    def decompress_and_unpack(self, split_info_package_list, encrypter):

        decompressor = PackedGHDecompressor(encrypter)
        split_info_list = decompressor.unpack_split_info(split_info_package_list)
        for split_info in split_info_list:
            g, h = unpack(split_info.sum_grad, self.offset, self.g_modulo, self.g_max_int, self.h_modulo, self.h_max_int,
                          self.exponent)
            split_info.sum_grad = g - split_info.sample_count * G_OFFSET
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
            rs_list.extend(p.unpack(self.encrypter))

        return rs_list


if __name__ == '__main__':

    from federatedml.secureprotol.fixedpoint import FixedPointNumber
    from federatedml.secureprotol import PaillierEncrypt, IterativeAffineEncrypt
    from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
    import numpy as np
    import time

    sample_num = 1000
    g = np.concatenate([-np.random.random(sample_num)])
    h = np.random.random(sample_num)

    encrypter = IterativeAffineEncrypt()
    encrypter.generate_key(1024)
    precision = 2 ** 53
    pos_max, neg_min = get_homo_encryption_max_int(encrypter)
    g_assign_bit, g_modulo, g_max_int, h_assign_bit, h_modulo, h_max_int, capacity = bit_assign_suggest(pos_max, neg_min,
                                                                                              sample_num, precision)

    s = time.time()
    print('g assign {} h assign {}'.format(g_assign_bit, h_assign_bit))

    exponent = FixedPointNumber.encode(0, g_modulo, g_max_int, precision).exponent
    pack_time_s = time.time()
    pack_g_h = []
    mul = pow(FixedPointNumber.BASE, exponent)

    for g_, h_ in zip(g, h):
        pack_g_h.append(ghpack((g_, h_), mul, g_modulo, h_modulo, offset=h_assign_bit))
        # pack_g_h.append(pack((g_, h_), g_modulo, g_max_int, h_modulo, h_max_int, h_assign_bit))

    pack_time_e = time.time()
    print('pack time', pack_time_e - pack_time_s)
    en_paillier = [raw_encrypt(i, encrypter, exponent) for i in pack_g_h]
    en_test = en_paillier[0]
    for i in en_paillier[1:500]:
        en_test += i
    en_test2 = en_paillier[500]
    for i in en_paillier[501:]:
        en_test2 += i

    g_sum_1, g_sum_2 = np.sum(g[0:500]), np.sum(g[500:])
    h_sum_1, h_sum_2 = np.sum(h[0:500]), np.sum(h[500:])

    print(g_sum_1, h_sum_1)
    print(g_sum_2, h_sum_2)
    de_rs = raw_decrypt(en_test, encrypter)
    test_g_1, test_h_1 = unpack(de_rs, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent)
    test_g_1 = test_g_1 - 500*G_OFFSET
    print(test_g_1, test_h_1)

    split_info_1 = SplitInfo(sum_grad=en_test, sample_count=500)
    split_info_2 = SplitInfo(sum_grad=en_test2, sample_count=500)
    pack = SplitInfoPackage(h_assign_bit + g_assign_bit, capacity, 0)
    pack.add(split_info_1)
    pack.add(split_info_2)


    # de_rs = raw_decrypt(en_test, encrypter)
    #
    # print(unpack(de_rs, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent))
    # e = time.time()
    # print('take time {}'.format(e - s))
    #
    # s = time.time()
    # en_g = [encrypter.encrypt(i) for i in g]
    # en_h = [encrypter.encrypt(i) for i in h]
    # g_rs = en_g[0]
    # for i in en_g[1:]:
    #     g_rs += i
    # h_rs = en_h[0]
    # for i in en_h[1:]:
    #     h_rs += i
    #
    # de_g = encrypter.decrypt(g_rs)
    # de_h = encrypter.decrypt(h_rs)
    # print(de_g)
    # print(de_h)
    # e = time.time()
    # print('take time {}'.format(e - s))
    #
    # print(g.sum())
    # print(h.sum())