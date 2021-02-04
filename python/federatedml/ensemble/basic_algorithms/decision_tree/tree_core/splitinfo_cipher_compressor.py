from federatedml.cipher_compressor.compressor import CipherCompressor, NormalCipherPackage, CipherDecompressor
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo


def get_g_h_info(task_type, max_sample_weight):

    if task_type == consts.CLASSIFICATION:
        g_offset, h_offset = 1, 0
        g_max, h_max = 2, 1
        return g_offset*max_sample_weight, h_offset*max_sample_weight, g_max*max_sample_weight, h_max*max_sample_weight
    else:
        raise ValueError('task type: {} is not supported by cipher compressing'.format(task_type))


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
        en_h = split_info.sum_hess
        super(SplitInfoPackage, self).add(en_g)
        super(SplitInfoPackage, self).add(en_h)
        self._cur_splitinfo_contains += 1
        self._split_info_without_gh.append(split_info_cp)

    def has_space(self):
        return self._capacity_left - 2 >= 0  # g and h

    def unpack(self, decrypter):
        unpack_rs = super(SplitInfoPackage, self).unpack(decrypter)
        g_list, h_list = unpack_rs[0::2], unpack_rs[1::2]
        for split_info, g, h in zip(self._split_info_without_gh, g_list, h_list):
            split_info.sum_grad = g
            split_info.sum_hess = h

        return self._split_info_without_gh


class GuestGradHessEncoder(object):

    def __init__(self, encrypter, encrypt_mode_calculator,
                 task_type=consts.CLASSIFICATION, round_decimal=7, max_sample_weights=1):

        self.max_sample_weights = max_sample_weights
        self.round_decimal = round_decimal
        self.g_offset, self.h_offset, self.g_max, self.h_max = get_g_h_info(task_type, max_sample_weights)

        self.encrypter = encrypter
        self.encrypt_mode_calculator = encrypt_mode_calculator

    def encode_and_encrypt(self, plain_text):
        return self.encrypter.encrypt(int(plain_text * 10**self.round_decimal))

    def encode_g_h_and_encrypt(self, g_h_table):

        g_offset, h_offset = self.g_offset, self.h_offset
        decimal_keeping_num = (10**self.round_decimal)
        g_h_table = g_h_table.mapValues(lambda x: (int((x[0]+g_offset)*decimal_keeping_num),
                                                   int((x[1]+h_offset)*decimal_keeping_num)))
        return self.encrypt_mode_calculator.encrypt(g_h_table)


class GuestSplitInfoDecompressor(object):

    def __init__(self, encrypter, task_type=consts.CLASSIFICATION, max_sample_weight=1):
        self.encrypter = encrypter
        self.decompressor = {}
        self.g_offset, self.h_offset, self.g_max, self.h_max = get_g_h_info(task_type, max_sample_weight)

    def renew_decompressor(self, node_map):

        self.decompressor = {}  # initialize new decompressors
        for node_id, idx in node_map.items():
            self.decompressor[node_id] = CipherDecompressor(self.encrypter)

    def unpack_split_info(self, node_id, packages):

        unpack_lists = self.decompressor[node_id].unpack(packages)
        rs = []
        for l_ in unpack_lists:
            rs += l_

        for split_info in rs:
            split_info.sum_grad = split_info.sum_grad - split_info.sample_count * self.g_offset
            split_info.sum_hess = split_info.sum_hess - split_info.sample_count * self.h_offset

        return rs


class HostSplitInfoCompressor(object):

    def __init__(self, max_capacity_int, encrypt_type, task_type=consts.CLASSIFICATION,
                 package_class=SplitInfoPackage, round_decimal=7, max_sample_weights=1):

        self.max_capacity_int = max_capacity_int
        self.encrypt_type = encrypt_type
        self.round_decimal = round_decimal
        self.max_sample_weights = max_sample_weights
        self.package_class = package_class
        self.compressors = {}
        self.g_offset, self.h_offset, self.g_max, self.h_max = get_g_h_info(task_type, max_sample_weights)

    def renew_compressor(self, node_sample_count, node_map):

        for node_id, idx in node_map.items():
            sample_num = node_sample_count[idx]
            max_float = sample_num*(max(self.g_max, self.h_max))
            self.compressors[node_id] = CipherCompressor(self.encrypt_type, max_capacity_int=self.max_capacity_int,
                                                         package_class=self.package_class,
                                                         round_decimal=self.round_decimal,
                                                         max_float=max_float)
            _, capacity = CipherCompressor.advise(max_float, self.max_capacity_int, self.encrypt_type, self.round_decimal)
            LOGGER.debug('compressor info of node {}: sample num {}, max capacity of a package {}'
                         ', max_float is {}'.format(node_id, sample_num, capacity, max_float))

    def compress_split_info(self, node_id, split_info_list, g_h_sum_info):
        split_info_list.append(g_h_sum_info)  # append to end
        packages = self.compressors[node_id].compress(split_info_list)
        return packages


if __name__ == '__main__':

    import numpy as np
    from federatedml.secureprotol import PaillierEncrypt as Encrypt
    from federatedml.secureprotol import IterativeAffineEncrypt as Encrypt

    def random_split_info_generate(num=5, max_num=90000):

        split_info_list = []
        for i in range(num):
            g, h = np.random.randint(max_num) + np.random.random(), np.random.randint(max_num) + np.random.random()
            best_fid, best_bid = np.random.randint(10), np.random.randint(10)
            missing_dir = np.random.randint(10000)
            info = SplitInfo(sum_grad=g, sum_hess=h, best_fid=best_fid, best_bid=best_bid, missing_dir=missing_dir,
                             sample_count=0)
            split_info_list.append(info)

        return split_info_list

    def en_split_info(en, split_info_list, decimal_to_keep):

        for s in split_info_list:
            plain_list.append(int(s.sum_grad * 10 ** decimal_to_keep))
            plain_list.append(int(s.sum_hess * 10 ** decimal_to_keep))
            s.sum_grad = en.encode_and_encrypt(s.sum_grad)
            s.sum_hess = en.encode_and_encrypt(s.sum_hess)

    def test_padding_num(plain_list, padding_num):
        rs_num = plain_list[0]
        for i in plain_list[1:]:
            rs_num = rs_num*padding_num + i
        return rs_num

    plain_list = []
    decimal_to_keep = 7
    key_length = 1024

    en = Encrypt()
    en.generate_key(key_length)

    encoder = GuestGradHessEncoder(en, None, )
    compressor = HostSplitInfoCompressor(key_length, consts.ITERATIVEAFFINE, )
    decompressor = GuestSplitInfoDecompressor(en, )

    compressor.renew_compressor([100000], {0: 0})
    decompressor.renew_decompressor({0: 0})

    gen_split_info = random_split_info_generate(num=10)
    print(gen_split_info)
    en_split_info(encoder, gen_split_info, decimal_to_keep)

    compressed_rs = compressor.compress_split_info(0, gen_split_info[:-1], gen_split_info[-1])
    rs = decompressor.unpack_split_info(0, compressed_rs)
    print(rs)

    # compressor = SplitInfoCompressor(key_length, round_decimal=7, task_type=consts.CLASSIFICATION)
    # compressor.init_encrypter(en, None)
    # compressor.renew_compressors([100000], {0: 0})
    #
    # gen_split_info = random_split_info_generate(num=30)
    # print(gen_split_info)
    # en_split_info(compressor, gen_split_info, decimal_to_keep)
    #
    # compressed_rs = compressor.compress_split_info(0, gen_split_info[:-1], gen_split_info[-1])
    # rs = compressor.unpack_split_info(0, compressed_rs)
    # print(rs)


