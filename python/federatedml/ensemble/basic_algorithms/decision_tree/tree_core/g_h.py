from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
import math

VERY_SMALL_FLOAT = -0.000000000001  # for computing edge cases


def suggest_parameters(paillier_max_int: int, sample_num: int, precision=2**53, g_max=1.0, g_min=-1.0, h_max=1.0):

    p_bit_len = paillier_max_int.bit_length()
    # make sure that combined plaintext is always smaller than paillier max_int
    available_bit = p_bit_len - 1
    # h is always larger than 0, compute bits fix point number needed
    h_max_int = FixedPointNumber.encode(h_max * sample_num + 1, precision=precision).encoding
    h_modulo = (h_max_int+1) * 3
    h_n_bit_len = h_modulo.bit_length()
    # in case fixed point num overflow, compute the bit length of edge case: max_encoding_num * sample_num
    h_max_encoding_edge_int = FixedPointNumber.encode(h_max, precision=precision).encoding * sample_num
    h_max_encoding_edge_int_bit_len = h_max_encoding_edge_int.bit_length()
    h_assign_bit = 0
    if h_max_encoding_edge_int_bit_len > h_n_bit_len:
        h_assign_bit = h_max_encoding_edge_int_bit_len + 1
    else:
        h_assign_bit = h_n_bit_len + 1
    assert h_assign_bit < available_bit, 'paillier key length is too small to conduct h packing'
    print('rs {} {} {}'.format(h_n_bit_len, h_max_encoding_edge_int_bit_len, h_assign_bit))
    # g contains positive and negative numbers, need to take negative case into consideration
    available_bit = available_bit - h_assign_bit
    assert available_bit > 0, 'no bit left for g packing'
    print('available bit {}'.format(available_bit))
    # for the ease of bit assignment, we reserve 1/3 as buffer
    buffer_bit = available_bit // 3
    g_assign_bit = available_bit - buffer_bit
    g_modulo = 1 << g_assign_bit
    g_max_int = (g_modulo // 3) - 1
    helper = FixedPointNumber.encode(0, g_modulo, g_max_int, precision)
    helper.encoding = helper.max_int
    fix_point_pos_max = helper.decode()
    helper.encoding = helper.n - helper.max_int
    fix_point_neg_min = helper.decode()
    assert g_max * sample_num < fix_point_pos_max, 'g assign bit not able to hold largest num'
    assert g_min * sample_num > fix_point_neg_min, 'g assign bit not able to hold smallest num'
    small_float = FixedPointNumber.encode(VERY_SMALL_FLOAT, g_modulo, g_max_int, precision)
    print(small_float.encoding)
    neg_int_edge_case = (small_float.encoding * sample_num)
    print('small encoding {}'.format(small_float.encoding))
    print('neg edge bit len {}'.format(neg_int_edge_case.bit_length()))
    assert neg_int_edge_case.bit_length() < available_bit, 'buffer bit length is not enough for edge case'
    print('g bit len {} h bit len {}'.format(g_assign_bit, h_assign_bit))

    return [g_assign_bit, g_modulo, g_max_int, h_assign_bit, h_modulo, h_max_int]


def pack(gh, g_modulo, g_max_int, h_modulo, h_max_int, offset, precision=2**53):

    g, h = gh[0], gh[1]
    g_encoding = FixedPointNumber.encode(g, g_modulo, g_max_int, precision).encoding
    h_encoding = FixedPointNumber.encode(h, h_modulo, h_max_int, precision).encoding
    pack_num = (g_encoding << offset) + h_encoding

    return pack_num


def pack_and_encrypt(gh, g_modulo, g_max_int, h_modulo, h_max_int, offset, encrypter, precision=2**53):

    pack_num = pack(gh, g_modulo, g_max_int, h_modulo, h_max_int, offset, precision)
    exponent = FixedPointNumber.encode(0, precision=precision).exponent
    paillier_num = raw_encrypt(pack_num, encrypter, exponent=exponent)
    return paillier_num, 0


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

    ciphertext = encrypter.public_key.raw_encrypt(plaintext)
    paillier_num = PaillierEncryptedNumber(encrypter.public_key, ciphertext, exponent)
    return paillier_num


def raw_decrypt(paillier_num, encrypter):

    decrypt_rs = encrypter.privacy_key.raw_decrypt(paillier_num.ciphertext())
    return decrypt_rs


class GHPacker(object):

    def __init__(self, paillier_max_int: int, sample_num: int, precision=2**53, g_max=1.0, g_min=-1.0, h_max=1.0):
        _, self.g_modulo, self.g_max_int, self.offset, self.h_modulo, self.h_max_int = \
            suggest_parameters(paillier_max_int, sample_num, precision, g_max, g_min, h_max)
        self.exponent = FixedPointNumber.encode(0, precision=precision).exponent
        self.precision = precision

    def pack(self, gh, encrypter):

        paillier_num = pack_and_encrypt(gh, self.g_modulo, self.g_max_int, self.h_modulo, self.h_max_int, self.offset, encrypter,
                                        self.precision)
        return paillier_num

    def unpack(self, paillier_num, encrypter):

        de_rs = raw_decrypt(paillier_num, encrypter)
        g, h = unpack(de_rs, self.offset, self.g_modulo, self.g_max_int, self.h_modulo, self.h_max_int,
                      self.exponent)
        return g, h


if __name__ == '__main__':

    from federatedml.secureprotol.fixedpoint import FixedPointNumber
    from federatedml.secureprotol import PaillierEncrypt
    import numpy as np
    import time

    sample_num = 100
    g = np.concatenate([-np.random.random(sample_num)])
    h = np.random.random(sample_num)

    RESERVED_BIT = 1
    encrypter = PaillierEncrypt()
    encrypter.generate_key(1024)
    precision = 2**53
    g_assign_bit, g_modulo, g_max_int, h_assign_bit, h_modulo, h_max_int = \
        suggest_parameters(encrypter.public_key.max_int, sample_num, precision=precision)

    s = time.time()
    print('g assign {} h assign {}'.format(g_assign_bit, h_assign_bit))
    exponent = FixedPointNumber.encode(0, g_modulo, g_max_int, precision).exponent

    pack_g_h = []
    for g_, h_ in zip(g, h):
        pack_g_h.append(pack((g_, h_), g_modulo, g_max_int, h_modulo, h_max_int, h_assign_bit, precision))

    en_paillier = [raw_encrypt(i, encrypter, exponent) for i in pack_g_h]
    en_test = en_paillier[0]
    for i in en_paillier[1:]:
        en_test += i
    de_rs = raw_decrypt(en_test, encrypter)

    print(g.sum())
    print(h.sum())
    unpack(de_rs, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent)
    e = time.time()
    print('take time {}'.format(e-s))

    s = time.time()
    en_g = [encrypter.encrypt(i) for i in g]
    en_h = [encrypter.encrypt(i) for i in h]
    g_rs = en_g[0]
    for i in en_g[1:]:
        g_rs += i
    h_rs = en_h[0]
    for i in en_h[1:]:
        h_rs += i
    de_g = encrypter.decrypt(g_rs)
    de_h = encrypter.decrypt(h_rs)
    print(de_g)
    print(de_h)
    e = time.time()
    print('take time {}'.format(e-s))
