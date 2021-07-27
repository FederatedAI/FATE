import functools
from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.cipher_compressor.packer import GuestIntegerPacker, cipher_list_to_cipher_tensor
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
from federatedml.util import consts
from federatedml.cipher_compressor.compressor import CipherCompressorHost, NormalCipherPackage
from federatedml.util import LOGGER

fix_point_precision = 2**52
REGRESSION_MAX_GRADIENT = 10**9


def post_func(x):
    # add a 0 to occupy h position
    return x[0], 0


def g_h_recover_post_func(unpack_rs_list: list, precision):
    if len(unpack_rs_list) == 2:
        g = unpack_rs_list[0] / precision
        h = unpack_rs_list[1] / precision
        return g, h
    else:
        return None, None


class SplitInfoPackage(NormalCipherPackage):

    def __init__(self, padding_length, max_capacity):
        super(SplitInfoPackage, self).__init__(padding_length, max_capacity)
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

    def unpack(self, decrypter):
        unpack_rs = super(SplitInfoPackage, self).unpack(decrypter)
        for split_info, g_h in zip(self._split_info_without_gh, unpack_rs):
            split_info.sum_grad = g_h

        return self._split_info_without_gh


class GHPacker(object):

    def __init__(self, sample_num: int, en_calculator: EncryptModeCalculator,
                       precision=fix_point_precision, max_sample_weight=1.0, task_type=consts.CLASSIFICATION,
                       g_min=None, g_max=None, sync_para=True):

        if task_type == consts.CLASSIFICATION:
            g_max = 1.0
            g_min = -1.0
            h_max = 1.0
        elif task_type == consts.REGRESSION:

            if g_min is None and g_max is None:
                g_max = REGRESSION_MAX_GRADIENT  # assign a large value for regression gradients
                g_min = -g_max
            else:
                g_max = g_max
                g_min = g_min
            h_max = 2.0
        else:
            raise ValueError('unknown task type {}'.format(task_type))

        self.g_max, self.g_min, self.h_max = g_max * max_sample_weight, g_min * max_sample_weight, h_max * max_sample_weight
        self.g_offset = abs(self.g_min)
        self.g_max_int, self.h_max_int = self._compute_packing_parameter(sample_num, precision)
        self.exponent = FixedPointNumber.encode(0, precision=precision).exponent
        self.precision = precision
        self.packer = GuestIntegerPacker(2, [self.g_max_int, self.h_max_int], encrypt_mode_calculator=en_calculator,
                                         sync_para=sync_para)

    def _compute_packing_parameter(self, sample_num: int, precision=2 ** 53):

        h_sum_max = self.h_max * sample_num
        h_max_int = int(h_sum_max * precision) + 1

        g_offset_max = self.g_offset + self.g_max
        g_max_int = int(g_offset_max * sample_num * precision) + 1

        return g_max_int, h_max_int

    @staticmethod
    def fixedpoint_encode(num, mul):
        int_fixpoint = int(round(num * mul))
        return int_fixpoint

    @staticmethod
    def to_fixedpoint(gh, mul, g_offset):

        g, h = gh[0], gh[1]
        g += g_offset  # to positive
        g_encoding = GHPacker.fixedpoint_encode(g, mul)
        h_encoding = GHPacker.fixedpoint_encode(h, mul)
        return g_encoding, h_encoding

    def pack_and_encrypt(self, gh):

        fixed_int_encode_func = functools.partial(self.to_fixedpoint, mul=self.precision, g_offset=self.g_offset)
        large_int_gh = gh.mapValues(fixed_int_encode_func)
        en_g_h = self.packer.pack_and_encrypt(large_int_gh, post_process_func=post_func)
        return en_g_h

    def decompress_and_unpack(self, split_info_package_list):

        rs = self.packer.decrypt_cipher_packages(split_info_package_list)
        for split_info in rs:
            g, h = g_h_recover_post_func(self.packer._unpack_an_int(split_info.sum_grad, self.packer._bit_assignment[0]),
                                         precision=self.precision)
            split_info.sum_grad = g - self.g_offset * split_info.sample_count
            split_info.sum_hess = h

        return rs


class PackedGHCompressor(object):

    def __init__(self, sync_para=True):
        self.compressor = CipherCompressorHost(package_class=SplitInfoPackage, sync_para=sync_para)

    def compress_split_info(self, split_info_list, g_h_sum_info):

        split_info_list.append(g_h_sum_info)  # append to end
        rs = self.compressor.compress(split_info_list)
        return rs

