#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import unittest
import functools
import math
from fate_arch.session import computing_session as session
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.g_h_optim import PackedGHCompressor, GHPacker, fix_point_precision
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
from federatedml.util import consts
import numpy as np


np.random.seed(114514)


def generate_bin_gh(num):
    # (-1, 1)
    g = np.random.random(num)
    h = np.random.random(num)
    g = g * 2 - 1
    return g, h


def generate_reg_gh(num, lower, upper):
    g = np.random.random(num)
    h = np.zeros(num) + 2
    g = g * (upper - lower) + lower
    return g, h


def cmp(a, b):
    if a[0] > b[0]:
        return 1
    else:
        return -1


def en_gh_list(g, h, en):
    en_g = [en.encrypt(i) for i in g]
    en_h = [en.encrypt(i) for i in h]
    return en_g, en_h


def truncate(f, n=consts.TREE_DECIMAL_ROUND):
    return math.floor(f * 10 ** n) / 10 ** n


def make_random_sum(collected_gh, g, h, en_g_l, en_h_l, max_sample_num):
    selected_sample_num = np.random.randint(max_sample_num) + 1  # at least 1 sample

    idx = np.random.random(selected_sample_num)
    idx = np.unique((idx * max_sample_num).astype(int))
    print('randomly select {} samples'.format(len(idx)))
    selected_g = g[idx]
    selected_h = h[idx]
    g_sum = selected_g.sum()
    h_sum = selected_h.sum()
    g_h_list = sorted(collected_gh, key=functools.cmp_to_key(cmp))
    sum_gh = 0
    en_g_sum = 0
    en_h_sum = 0
    for i in idx:
        gh = g_h_list[i][1][0]

        sum_gh += gh
        en_g_sum += en_g_l[i]
        en_h_sum += en_h_l[i]

    return g_sum, h_sum, sum_gh, en_g_sum, en_h_sum, len(idx)


class TestFeatureHistogram(unittest.TestCase):

    @staticmethod
    def prepare_testing_data(g, h, en, max_sample_num, sample_id, task_type, g_min=None, g_max=None):

        packer = GHPacker(max_sample_num, encrypter=en, sync_para=False, task_type=task_type,
                          g_min=g_min, g_max=g_max)
        en_g_l, en_h_l = en_gh_list(g, h, en)
        data_list = [(id_, (g_, h_)) for id_, g_, h_ in zip(sample_id, g, h)]
        data_table = session.parallelize(data_list, 4, include_key=True)
        en_table = packer.pack_and_encrypt(data_table)
        collected_gh = list(en_table.collect())

        return packer, en_g_l, en_h_l, en_table, collected_gh

    @classmethod
    def setUpClass(cls):

        session.init("test_gh_packing")

        cls.max_sample_num = 1000
        cls.test_num = 10
        cls.split_info_test_num = 200

        key_length = 1024
        sample_id = [i for i in range(cls.max_sample_num)]

        # classification data
        cls.g, cls.h = generate_bin_gh(cls.max_sample_num)

        cls.p_en = PaillierEncrypt()
        cls.p_en.generate_key(key_length)

        cls.p_packer, cls.p_en_g_l, cls.p_en_h_l, cls.p_en_table, cls.p_collected_gh = \
            cls.prepare_testing_data(cls.g, cls.h, cls.p_en, cls.max_sample_num, sample_id, consts.CLASSIFICATION)
        cls.compressor = PackedGHCompressor(sync_para=False)
        cls.compressor.compressor._padding_length, cls.compressor.compressor._capacity = \
            cls.p_packer.packer.cipher_compress_suggest()
        print('paillier compress para {}'.format(cls.p_packer.packer.cipher_compress_suggest()))

        # regression data
        cls.g_reg, cls.h_reg = generate_reg_gh(cls.max_sample_num, -1000, 1000)
        cls.reg_p_packer, cls.reg_p_en_g_l, cls.reg_p_en_h_l, cls.reg_p_en_table, cls.reg_p_collected_gh = \
            cls.prepare_testing_data(cls.g_reg, cls.h_reg, cls.p_en, cls.max_sample_num, sample_id, consts.REGRESSION,
                                     g_min=-1000, g_max=1000)
        cls.reg_compressor = PackedGHCompressor(sync_para=False)
        cls.reg_compressor.compressor._padding_length, cls.reg_compressor.compressor._capacity = \
            cls.reg_p_packer.packer.cipher_compress_suggest()
        print('paillier compress para {}'.format(cls.p_packer.packer.cipher_compress_suggest()))

        print('initialization done')

    def run_gh_accumulate_test(self, test_num, collected_gh, en_g_l, en_h_l, packer, en, g, h, check=True):

        print('{} test to run'.format(test_num))
        for i in range(test_num):
            print('executing test {}'.format(i))
            g_sum, h_sum, en_sum, en_g_sum, en_h_sum, sample_num = make_random_sum(collected_gh, g, h,
                                                                                   en_g_l,
                                                                                   en_h_l,
                                                                                   self.max_sample_num)
            de_num = en.raw_decrypt(en_sum)
            unpack_num = packer.packer.unpack_an_int(de_num, packer.packer.bit_assignment[0])

            g_sum_ = unpack_num[0] / fix_point_precision - sample_num * packer.g_offset
            h_sum_ = unpack_num[1] / fix_point_precision

            g_sum_2 = en.decrypt(en_g_sum)
            h_sum_2 = en.decrypt(en_h_sum)

            print(g_sum, h_sum)
            print(g_sum_2, h_sum_2)
            print(g_sum_, h_sum_)

            g_sum, h_sum = truncate(g_sum), truncate(h_sum)
            g_sum_, h_sum_ = truncate(g_sum_), truncate(h_sum_)
            g_sum_2, h_sum_2 = truncate(g_sum_2), truncate(h_sum_2)

            print(g_sum, h_sum)
            print(g_sum_2, h_sum_2)
            print(g_sum_, h_sum_)

            if check:
                # make sure packing result close to plaintext sum
                self.assertTrue(g_sum_ == g_sum)
                self.assertTrue(h_sum_ == h_sum)

            print('passed')

    def test_pack_gh_accumulate(self):

        # test the correctness of gh packing(in comparision to plaintext)

        # Paillier
        self.run_gh_accumulate_test(self.test_num, self.p_collected_gh, self.p_en_g_l, self.p_en_h_l, self.p_packer,
                                    self.p_en, self.g, self.h)

        print('*' * 30)
        print('test paillier done')
        print('*' * 30)

    def test_split_info_cipher_compress(self):

        # test the correctness of cipher compressing
        print('testing binary')
        collected_gh = self.p_collected_gh
        en_g_l = self.p_en_g_l
        en_h_l = self.p_en_h_l
        packer = self.p_packer
        en = self.p_en

        sp_list = []
        g_sum_list, h_sum_list = [], []
        pack_en_list = []

        for i in range(self.split_info_test_num):
            g_sum, h_sum, en_sum, en_g_sum, en_h_sum, sample_num = make_random_sum(collected_gh, self.g, self.h,
                                                                                   en_g_l,
                                                                                   en_h_l,
                                                                                   self.max_sample_num)
            sp = SplitInfo(sum_grad=en_sum, sum_hess=0, sample_count=sample_num)
            sp_list.append(sp)
            g_sum_list.append(g_sum)
            h_sum_list.append(h_sum)
            pack_en_list.append(en_sum)

        print('generating split-info done')
        packages = self.compressor.compress_split_info(sp_list[:-1], sp_list[-1])
        print('package length is {}'.format(len(packages)))
        unpack_rs = packer.decompress_and_unpack(packages)
        case_id = 0
        for s, g, h, en_gh in zip(unpack_rs, g_sum_list, h_sum_list, pack_en_list):
            print('*' * 10)
            print(case_id)
            case_id += 1
            de_num = en.raw_decrypt(en_gh)
            unpack_num = packer.packer.unpack_an_int(de_num, packer.packer.bit_assignment[0])
            g_sum_ = unpack_num[0] / fix_point_precision - s.sample_count * packer.g_offset
            h_sum_ = unpack_num[1] / fix_point_precision

            print(s.sample_count)
            print(s.sum_grad, g_sum_, g)
            print(s.sum_hess, h_sum_, h)

            # make sure cipher compress is correct
            self.assertTrue(truncate(s.sum_grad) == truncate(g_sum_))
            self.assertTrue(truncate(s.sum_hess) == truncate(h_sum_))
        print('check passed')

    def test_regression_cipher_compress(self):

        # test the correctness of cipher compressing
        print('testing regression')
        collected_gh = self.reg_p_collected_gh
        en_g_l = self.reg_p_en_g_l
        en_h_l = self.reg_p_en_h_l
        packer = self.reg_p_packer
        en = self.p_en

        sp_list = []
        g_sum_list, h_sum_list = [], []
        pack_en_list = []

        for i in range(self.split_info_test_num):
            g_sum, h_sum, en_sum, en_g_sum, en_h_sum, sample_num = make_random_sum(collected_gh, self.g_reg, self.h_reg,
                                                                                   en_g_l,
                                                                                   en_h_l,
                                                                                   self.max_sample_num)
            sp = SplitInfo(sum_grad=en_sum, sum_hess=0, sample_count=sample_num)
            sp_list.append(sp)
            g_sum_list.append(g_sum)
            h_sum_list.append(h_sum)
            pack_en_list.append(en_sum)

        print('generating split-info done')
        packages = self.reg_compressor.compress_split_info(sp_list[:-1], sp_list[-1])
        print('package length is {}'.format(len(packages)))
        unpack_rs = packer.decompress_and_unpack(packages)
        case_id = 0
        for s, g, h, en_gh in zip(unpack_rs, g_sum_list, h_sum_list, pack_en_list):
            print('*' * 10)
            print(case_id)
            case_id += 1
            de_num = en.raw_decrypt(en_gh)  # make sure packing result close to plaintext sum
            unpack_num = packer.packer.unpack_an_int(de_num, packer.packer.bit_assignment[0])
            g_sum_ = unpack_num[0] / fix_point_precision - s.sample_count * packer.g_offset
            h_sum_ = unpack_num[1] / fix_point_precision

            print(s.sample_count)
            print(s.sum_grad, g_sum_, g)
            print(s.sum_hess, h_sum_, h)

            # make sure cipher compress is correct
            self.assertTrue(truncate(s.sum_grad) == truncate(g_sum_))
            self.assertTrue(truncate(s.sum_hess) == truncate(h_sum_))
        print('check passed')

    def test_regression_gh_packing(self):

        # Paillier
        self.run_gh_accumulate_test(
            self.test_num,
            self.reg_p_collected_gh,
            self.reg_p_en_g_l,
            self.reg_p_en_h_l,
            self.reg_p_packer,
            self.p_en,
            self.g_reg,
            self.h_reg,
            check=False)  # float error in regression is not controllable

    @classmethod
    def tearDownClass(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
