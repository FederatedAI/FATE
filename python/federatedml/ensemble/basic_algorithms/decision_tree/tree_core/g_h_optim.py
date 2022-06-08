import functools
import numpy as np
from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import PaillierEncrypt
from federatedml.cipher_compressor.packer import GuestIntegerPacker, cipher_list_to_cipher_tensor
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
from federatedml.util import consts
from federatedml.cipher_compressor.compressor import CipherCompressorHost, NormalCipherPackage
from federatedml.cipher_compressor.compressor import PackingCipherTensorPackage
from federatedml.util import LOGGER

fix_point_precision = 2 ** 52
REGRESSION_MAX_GRADIENT = 10 ** 9


def post_func(x):
    # add a 0 to occupy h position
    return x[0], 0


def g_h_recover_post_func(unpack_rs_list: list, precision):
    if len(unpack_rs_list) == 2:
        g = unpack_rs_list[0] / precision
        h = unpack_rs_list[1] / precision
        return g, h
    else:
        g_list, h_list = [], []
        for g, h in zip(unpack_rs_list[0::2], unpack_rs_list[1::2]):
            g_list.append(g / precision)
            h_list.append(h / precision)
        return np.array(g_list), np.array(h_list)


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


class SplitInfoPackage2(PackingCipherTensorPackage):

    def __init__(self, padding_length, max_capacity):
        super(SplitInfoPackage2, self).__init__(padding_length, max_capacity)
        self._split_info_without_gh = []
        self._cur_splitinfo_contains = 0

    def add(self, split_info):
        split_info_cp = SplitInfo(sitename=split_info.sitename, best_bid=split_info.best_bid,
                                  best_fid=split_info.best_fid, missing_dir=split_info.missing_dir,
                                  mask_id=split_info.mask_id, sample_count=split_info.sample_count)

        en_g = split_info.sum_grad
        super(SplitInfoPackage2, self).add(en_g)
        self._cur_splitinfo_contains += 1
        self._split_info_without_gh.append(split_info_cp)

    def unpack(self, decrypter):
        unpack_rs = super(SplitInfoPackage2, self).unpack(decrypter)
        for split_info, g_h in zip(self._split_info_without_gh, unpack_rs):
            split_info.sum_grad = g_h

        return self._split_info_without_gh


class GHPacker(object):

    def __init__(self, sample_num: int, encrypter: PaillierEncrypt,
                 precision=fix_point_precision, max_sample_weight=1.0, task_type=consts.CLASSIFICATION,
                 g_min=None, g_max=None, class_num=1, mo_mode=False, sync_para=True):

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
        self.class_num = class_num
        self.mo_mode = mo_mode
        self.packer = GuestIntegerPacker(class_num * 2, [self.g_max_int, self.h_max_int] * class_num,
                                         encrypter=encrypter,
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
    def to_fixedpoint_arr_format(gh, mul, g_offset):

        en_list = []
        g_arr, h_arr = gh
        for g, h in zip(g_arr, h_arr):
            g += g_offset  # to positive
            g_encoding = GHPacker.fixedpoint_encode(g, mul)
            h_encoding = GHPacker.fixedpoint_encode(h, mul)
            en_list.append(g_encoding)
            en_list.append(h_encoding)

        return en_list

    @staticmethod
    def to_fixedpoint(gh, mul, g_offset):
        g, h = gh
        return [GHPacker.fixedpoint_encode(g + g_offset, mul), GHPacker.fixedpoint_encode(h, mul)]

    def pack_and_encrypt(self, gh):

        fixedpoint_encode_func = self.to_fixedpoint
        if self.mo_mode:
            fixedpoint_encode_func = self.to_fixedpoint_arr_format
        fixed_int_encode_func = functools.partial(fixedpoint_encode_func, mul=self.precision, g_offset=self.g_offset)
        large_int_gh = gh.mapValues(fixed_int_encode_func)
        if not self.mo_mode:
            en_g_h = self.packer.pack_and_encrypt(large_int_gh,
                                                  post_process_func=post_func)  # take cipher out from list
        else:
            en_g_h = self.packer.pack_and_encrypt(large_int_gh)
            en_g_h = en_g_h.mapValues(lambda x: (x, 0))  # add 0 to occupy h position

        return en_g_h

    def decompress_and_unpack(self, split_info_package_list):

        rs = self.packer.decrypt_cipher_packages(split_info_package_list)
        for split_info in rs:
            if self.mo_mode:
                unpack_rs = self.packer.unpack_an_int_list(split_info.sum_grad)
            else:
                unpack_rs = self.packer.unpack_an_int(split_info.sum_grad, self.packer.bit_assignment[0])
            g, h = g_h_recover_post_func(unpack_rs, fix_point_precision)
            split_info.sum_grad = g - self.g_offset * split_info.sample_count
            split_info.sum_hess = h

        return rs


class PackedGHCompressor(object):

    def __init__(self, sync_para=True, mo_mode=False):
        package_class = SplitInfoPackage
        if mo_mode:
            package_class = SplitInfoPackage2
        self.compressor = CipherCompressorHost(package_class=package_class, sync_para=sync_para)

    def compress_split_info(self, split_info_list, g_h_sum_info):
        split_info_list.append(g_h_sum_info)  # append to end
        rs = self.compressor.compress(split_info_list)
        return rs
