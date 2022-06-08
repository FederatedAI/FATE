import functools
from federatedml.util import LOGGER
from federatedml.secureprotol import PaillierEncrypt
from federatedml.cipher_compressor.compressor import get_homo_encryption_max_int
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.cipher_compressor.compressor import PackingCipherTensor
from federatedml.cipher_compressor.compressor import CipherPackage
from federatedml.transfer_variable.transfer_class.cipher_compressor_transfer_variable \
    import CipherCompressorTransferVariable
from federatedml.util import consts


def cipher_list_to_cipher_tensor(cipher_list: list):
    cipher_tensor = PackingCipherTensor(ciphers=cipher_list)
    return cipher_tensor


class GuestIntegerPacker(object):

    def __init__(self, pack_num: int, pack_num_range: list, encrypter: PaillierEncrypt,
                 sync_para=True):
        """
        max_int: max int allowed for packing result
        pack_num: number of int to pack, they must be POSITIVE integer
        pack_num_range: list of integer, it gives range of every integer to pack
        need_cipher_compress: if dont need cipher compress, related parameter will be set to 1
        """

        self._pack_num = pack_num
        assert len(pack_num_range) == self._pack_num, 'list len must equal to pack_num'
        self._pack_num_range = pack_num_range
        self._pack_num_bit = [i.bit_length() for i in pack_num_range]
        self.encrypter = encrypter

        max_pos_int, _ = get_homo_encryption_max_int(self.encrypter)
        self._max_int = max_pos_int
        self._max_bit = self._max_int.bit_length() - 1  # reserve 1 bit, in case overflow

        # sometimes max_int is not able to hold all num need to be packed, so we
        # use more than one large integer to pack them all
        self.bit_assignment = []
        tmp_list = []
        bit_count = 0
        for bit_len in self._pack_num_bit:
            if bit_count + bit_len >= self._max_bit:
                if bit_count == 0:
                    raise ValueError('unable to pack this num using in current int capacity')
                self.bit_assignment.append(tmp_list)
                tmp_list = []
                bit_count = 0
            bit_count += bit_len
            tmp_list.append(bit_len)

        if len(tmp_list) != 0:
            self.bit_assignment.append(tmp_list)
        self._pack_int_needed = len(self.bit_assignment)

        # transfer variable
        compress_parameter = self.cipher_compress_suggest()

        if sync_para:
            self.trans_var = CipherCompressorTransferVariable()
            self.trans_var.compress_para.remote(compress_parameter, role=consts.HOST, idx=-1)

        LOGGER.debug('int packer init done, bit assign is {}, compress para is {}'.format(self.bit_assignment,
                                                                                          compress_parameter))

    def cipher_compress_suggest(self):
        compressible = self.bit_assignment[-1]
        total_bit_count = sum(compressible)
        compress_num = self._max_bit // total_bit_count
        padding_bit = total_bit_count
        return padding_bit, compress_num

    def pack_int_list(self, int_list: list):

        assert len(int_list) == self._pack_num, 'list length is not equal to pack_num'
        start_idx = 0
        rs = []
        for bit_assign_of_one_int in self.bit_assignment:
            to_pack = int_list[start_idx: start_idx + len(bit_assign_of_one_int)]
            packing_rs = self._pack_fix_len_int_list(to_pack, bit_assign_of_one_int)
            rs.append(packing_rs)
            start_idx += len(bit_assign_of_one_int)

        return rs

    def _pack_fix_len_int_list(self, int_list: list, bit_assign: list):

        result = int_list[0]
        for i, offset in zip(int_list[1:], bit_assign[1:]):
            result = result << offset
            result += i

        return result

    def unpack_an_int(self, integer: int, bit_assign_list: list):

        rs_list = []
        for bit_assign in reversed(bit_assign_list[1:]):
            mask_int = (2**bit_assign) - 1
            unpack_int = integer & mask_int
            rs_list.append(unpack_int)
            integer = integer >> bit_assign
        rs_list.append(integer)

        return list(reversed(rs_list))

    def pack(self, data_table):
        packing_data_table = data_table.mapValues(self.pack_int_list)
        return packing_data_table

    def pack_and_encrypt(self, data_table, post_process_func=cipher_list_to_cipher_tensor):

        packing_data_table = self.pack(data_table)
        en_packing_data_table = self.encrypter.distribute_raw_encrypt(packing_data_table)

        if post_process_func:
            en_packing_data_table = en_packing_data_table.mapValues(post_process_func)

        return en_packing_data_table

    def unpack_result(self, decrypted_result_list: list, post_func=None):

        final_rs = []
        for l_ in decrypted_result_list:

            rs_list = self.unpack_an_int_list(l_, post_func)
            final_rs.append(rs_list)

        return final_rs

    def unpack_an_int_list(self, int_list, post_func=None):

        assert len(int_list) == len(self.bit_assignment), 'length of integer list is not equal to bit_assignment'
        rs_list = []
        for idx, integer in enumerate(int_list):
            unpack_list = self.unpack_an_int(integer, self.bit_assignment[idx])
            if post_func:
                unpack_list = post_func(unpack_list)
            rs_list.extend(unpack_list)

        return rs_list

    def decrypt_cipher_packages(self, content):

        if isinstance(content, list):

            assert issubclass(type(content[0]), CipherPackage), 'content is not CipherPackages'
            decrypt_rs = []
            for i in content:
                unpack_ = i.unpack(self.encrypter)
                decrypt_rs += unpack_
            return decrypt_rs

        else:
            raise ValueError('illegal input type')

    def decrypt_cipher_package_and_unpack(self, data_table):

        de_func = functools.partial(self.decrypt_cipher_packages)
        de_table = data_table.mapValues(de_func)
        unpack_table = de_table.mapValues(self.unpack_result)

        return unpack_table
