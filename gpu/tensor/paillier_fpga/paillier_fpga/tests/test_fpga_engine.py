#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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
import numpy
import unittest
import random
import functools
import operator
import time

from fate_arch.tensor.impl.blocks.python_paillier_block import (
    PaillierKeypair,
    PaillierEncryptedNumber,
    FixedPointNumber,
)

from ..fpga_engine import (
    FLOAT_TYPE,
    INT64_TYPE,
    pi_p2c_pub_key,
    pi_p2c_priv_key,
    pi_h2d_pub_key,
    pi_h2d_priv_key,
    TensorShapeStorage,
    bi_alloc,
    PLAIN_BYTE,
    MEM_HOST,
    te_alloc,
    fp_alloc,
    pi_alloc,
    te_p2c,
    fp_encode,
    fp_decode,
    te_c2p,
    pi_encrypt,
    pi_gen_obf_seed,
    CIPHER_BITS,
    pi_obfuscate,
    pi_c2p,
    pi_decrypt,
    fp_mul,
    fp_c2p,
    pi_add,
    pi_mul,
    pi_sum,
    bi_free,
    te_free,
    fp_free,
    pi_free, te_slice, initialize_device, fp_p2c, pi_p2c, bi_gen_rand, bi_c2p, pi_transpose, pi_matmul, fp_transpose,
    CIPHER_BYTE, te_c2bytes, te_bytes2c, fp_c2bytes, fp_bytes2c, pi_c2bytes, pi_bytes2c, pi_slice, pi_reshape,
    te_c2p_first, TensorStorage, te_c2p_shape, te_cat, te_pow, te_add, te_mul, te_truediv, te_floordiv, te_sub,
    te_matmul, te_abs, te_transpose, te_reshape, te_exp, te_hstack, pi_cat, te_sum, fp_slice, te_p2c_shape, fp_cat,
    te_neg,
)

# SWITCH DATA TYPE HERE
# EITHER INT64_TYPE OR FLOAT_TYPE
RAND_TYPE = INT64_TYPE

TEST_SIZE = 6
ERROR_TOLERANCE = 1e-10


def generate_rand(test_size):
    if RAND_TYPE == FLOAT_TYPE:
        return numpy.random.normal(0, 10, test_size)
    elif RAND_TYPE == INT64_TYPE:
        return numpy.random.randint(-2 ** 30, 2 ** 30, test_size)
    else:
        raise TypeError("Invalid data type")


def assert_diff(res, ref):
    if res == 0 or ref == 0:
        assert res == 0
        assert ref == 0
    else:
        diff = res - ref
        assert abs(diff / res) < ERROR_TOLERANCE
        assert abs(diff / ref) < ERROR_TOLERANCE


def assert_ndarray_diff(res, ref):
    assert res.shape == ref.shape
    res, ref = res.flatten(), ref.flatten()
    assert res.shape == ref.shape
    for i in range(res.size):
        assert_diff(res[i], ref[i])


class TestOperators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # sys.stdout = open("stdout.log", 'a')  # uncomment this to redirect stdout
        # random.seed(time.time())  # no need to set random.seed as we're using numpy.random
        initialize_device()
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        cls.n, cls.max_int = cls._pub_key.n, cls._pub_key.max_int
        cls._cpu_pub_key = pi_p2c_pub_key(None, cls._pub_key)
        cls._cpu_priv_key = pi_p2c_priv_key(None, cls._priv_key)
        cls._fpga_pub_key = pi_h2d_pub_key(None, cls._cpu_pub_key)
        cls._fpga_priv_key = pi_h2d_priv_key(None, cls._cpu_priv_key)
        print("\n\n", "*" * 100, "\n\nInitialization complete\nTest Size:", TEST_SIZE)

    # test encode and decode
    # using operators: te_p2c, fp_encode, fp_c2p, fp_decode, te_c2p
    def test_encode_and_decode(self):
        print("\n\n", "*" * 100, "\n\nTest Encode and Decode Begins")

        raw = generate_rand(TEST_SIZE)
        raw[TEST_SIZE // 2] = 0  # test encode zero
        store = te_p2c(raw, None)
        precision = 10000 if RAND_TYPE == FLOAT_TYPE else None

        # check encoded numbers (fixed-point numbers)
        fpga_encoded_store = fp_encode(store, self.n, self.max_int, precision, None)
        fpga_encoded = fp_c2p(fpga_encoded_store)
        cpu_encoded = [FixedPointNumber.encode(v, self.n, self.max_int, precision) for v in raw]
        assert len(fpga_encoded) == TEST_SIZE
        assert len(cpu_encoded) == TEST_SIZE
        for i in range(TEST_SIZE):
            print("i:", i, ", raw data:", raw[i])
            print("FPGA encoding:", fpga_encoded[i].encoding, ", base:", fpga_encoded[i].BASE, ", exp:",
                  fpga_encoded[i].exponent)
            print("CPU encoding:", cpu_encoded[i].encoding, ", base:", cpu_encoded[i].BASE, ", exp:",
                  cpu_encoded[i].exponent)
        for i in range(TEST_SIZE):
            assert fpga_encoded[i].encoding == cpu_encoded[i].encoding
            assert fpga_encoded[i].BASE == cpu_encoded[i].BASE
            assert fpga_encoded[i].exponent == cpu_encoded[i].exponent

        # check decoded numbers
        cpu_encoded_cpu_decoded = [v.decode() for v in cpu_encoded]
        cpu_encoded_fpga_decoded = te_c2p(fp_decode(fp_p2c(None, cpu_encoded, RAND_TYPE), None, None))
        fpga_encoded_cpu_decoded = [v.decode() for v in fpga_encoded]
        fpga_encoded_fpga_decoded = te_c2p(fp_decode(fpga_encoded_store, None, None))
        assert len(cpu_encoded_cpu_decoded) == TEST_SIZE
        assert len(cpu_encoded_fpga_decoded) == TEST_SIZE
        assert len(fpga_encoded_cpu_decoded) == TEST_SIZE
        assert len(fpga_encoded_fpga_decoded) == TEST_SIZE
        for i in range(TEST_SIZE):
            print("decoded compare: i:", i, cpu_encoded_cpu_decoded[i], cpu_encoded_fpga_decoded[i],
                  fpga_encoded_cpu_decoded[i], fpga_encoded_fpga_decoded[i])
            assert_diff(cpu_encoded_fpga_decoded[i], cpu_encoded_cpu_decoded[i])
            assert_diff(fpga_encoded_cpu_decoded[i], cpu_encoded_cpu_decoded[i])
            assert_diff(fpga_encoded_fpga_decoded[i], cpu_encoded_cpu_decoded[i])

        print("test passed")

    # test encrypt and decrypt
    # using operators: fp_encode, pi_encrypt, pi_decrypt, te_p2c, te_c2p, pi_c2p
    def test_encrypt_and_decrypt(self):
        print("\n\n", "*" * 100, "\n\nTest Encrypt And Decrypt Begins")

        print("\nPart 1: FPGA encrypt, FPGA decrypt")
        raw = generate_rand(TEST_SIZE)
        store = te_p2c(raw, None)
        encoded = fp_encode(store, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded, None, None)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, encrypted, None, None)
        ref1 = te_c2p(decrypted)
        assert store.data_type == RAND_TYPE
        assert encoded.data_type == RAND_TYPE
        assert encrypted.data_type == RAND_TYPE
        assert decrypted.data_type == RAND_TYPE
        for i in range(TEST_SIZE):
            print("i:", i, ", original:", raw[i], ", decrypted:", ref1[i])
            assert_diff(raw[i], ref1[i])

        print("\nPart 2: FPGA encrypt, CPU decrypt")
        tmp_enc, _, tmp_exp = pi_c2p(encrypted)
        pen_recv = [PaillierEncryptedNumber(self._pub_key, tmp_enc[i], int(round(tmp_exp[i]))) for i in
                    range(TEST_SIZE)]
        ref2 = [self._priv_key.decrypt(v) for v in pen_recv]
        for i in range(TEST_SIZE):
            print("i:", i, ", original:", raw[i], ", decrypted:", ref2[i])
            assert_diff(raw[i], ref2[i])

        print("\nPart 3: CPU encrypt, FPGA decrypt")
        # print("FPGA decrypting a CPU encrypted number currently unavailable, needs pi_p2c support")
        cpu_encrypted = [self._pub_key.encrypt(raw[i], None, 0) for i in range(TEST_SIZE)]
        for i in range(TEST_SIZE):
            print("FPGA: i:", i, ", cipher text:", pen_recv[i].ciphertext(False), ", exp:", pen_recv[i].exponent)
            print("CPU: i:", i, ", cipher text:", cpu_encrypted[i].ciphertext(False), ", exp:",
                  cpu_encrypted[i].exponent)
            assert pen_recv[i].exponent == cpu_encrypted[i].exponent
            try:
                assert pen_recv[i].ciphertext(False) == cpu_encrypted[i].ciphertext(False)
            except AssertionError:
                # Note that there's an approx 1/1000 probability that these ciphers don't match
                # However, this shouldn't affect the final result
                print("\n>>>>>> The following cipher texts didn't match:")
                print("raw number:", raw[i])
                print("FPGA encoding:", fp_c2p(encoded)[i].encoding)
                print("CPU encoding:", FixedPointNumber.encode(raw[i], self.n, self.max_int).encoding)
                print("FPGA cipher:", pen_recv[i].ciphertext(False))
                print("CPU cipher:", cpu_encrypted[i].ciphertext(False))
                print("pub_key.n:", self._pub_key.n)
                print("pub_key.nsquare:", self._pub_key.nsquare)
                print("priv_key.p:", self._priv_key.p)
                print("priv_key.q:", self._priv_key.q)
                print(">>>>>> End Dumping\n")
        pi_store = pi_p2c(None, cpu_encrypted, RAND_TYPE)
        ref3 = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, pi_store, None, None))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], ref3[i])
            print("i:", i, ", original:", raw[i], ", decrypted:", ref3[i])

        print("test passed")

    def test_pi_add(self):
        print("\n\n", "*" * 100, "\n\nTest Paillier Encrypted Number Add Begins")
        raw_1, raw_2 = generate_rand(2), generate_rand(TEST_SIZE)
        te_store_1, te_store_2 = te_p2c(raw_1, None), te_p2c(raw_2, None)
        encoded_1, encoded_2 = fp_encode(te_store_1, self.n, self.max_int), fp_encode(te_store_2, self.n, self.max_int)
        encrypted_1, encrypted_2 = pi_encrypt(self._fpga_pub_key, encoded_1, None, None), pi_encrypt(self._fpga_pub_key,
                                                                                                     encoded_2, None,
                                                                                                     None)
        shape_1, shape_2 = TensorShapeStorage(2, 1), TensorShapeStorage(2, 3)  # passed different shapes
        res_store, res_shape = pi_add(self._fpga_pub_key, encrypted_1, encrypted_2, shape_1, shape_2, None, None, None)
        assert res_shape.to_tuple() == (2, 3)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, res_store, None, None)
        received = te_c2p(decrypted)
        for i in range(TEST_SIZE):
            print("i:", i, ", raw result:", raw_1[i // 3] + raw_2[i], ", FPGA result:", received[i])
            assert_diff(raw_1[i // 3] + raw_2[i], received[i])
        print("test passed")

    def test_pi_mul(self):
        print("\n\n", "*" * 100, "\n\nTest PEN Multiplies FPN Begins")
        raw_1, raw_2 = generate_rand(3), generate_rand(TEST_SIZE)
        te_store_1, te_store_2 = te_p2c(raw_1, None), te_p2c(raw_2, None)
        encoded_1, encoded_2 = fp_encode(te_store_1, self.n, self.max_int), fp_encode(te_store_2, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded_1, None, None)
        shape_1, shape_2 = TensorShapeStorage(3), TensorShapeStorage(2, 3)  # passed different shapes
        res_store, res_shape = pi_mul(self._fpga_pub_key, encrypted, encoded_2, shape_1, shape_2, None, None, None)
        assert res_shape.to_tuple() == (2, 3)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, res_store, None, None)
        received = te_c2p(decrypted)
        for i in range(TEST_SIZE):
            print("i:", i, ", raw result:", raw_1[i % 3] * raw_2[i], ", FPGA result:", received[i])
            assert_diff(raw_1[i % 3] * raw_2[i], received[i])
        print("test passed")

    def test_gen_obf_seed(self):
        print("\n\n", "*" * 100, "\n\nTest Generate Obfscator Begins")
        # why divided by 6, see pi_gen_obf_seed implementation
        bi_store = pi_gen_obf_seed(None, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, 0, None)
        obfuscators = bi_c2p(bi_store.bigint_storage, 0, TEST_SIZE)
        for i in range(TEST_SIZE):
            print("i:", i, "obfuscator:", obfuscators[i])
            assert CIPHER_BITS * 0.9 <= obfuscators[i].bit_length()
            assert obfuscators[i].bit_length() <= CIPHER_BITS
        print("test passed")

    def test_obfuscate(self):
        print("\n\n", "*" * 100, "\n\nTest Obfuscate Begins")

        # generate big random values
        bi_rand_store = bi_gen_rand(CIPHER_BITS // 6, TEST_SIZE, None, 0, None)
        bi_rand_vals = bi_c2p(bi_rand_store.bigint_storage, 0, TEST_SIZE)
        obf_rand_store = pi_gen_obf_seed(None, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, 0, None)

        print("\nPart 1: FPGA encrypt, FPGA decrypt")
        raw = generate_rand(TEST_SIZE)
        store = te_p2c(raw, None)
        encoded = fp_encode(store, self.n, self.max_int)
        raw_encrypted = pi_encrypt(self._fpga_pub_key, encoded, None, None)
        encrypted = pi_obfuscate(self._fpga_pub_key, raw_encrypted, obf_rand_store, None, None)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, encrypted, None, None)
        ref1 = te_c2p(decrypted)
        for i in range(TEST_SIZE):
            assert_diff(raw[i], ref1[i])
            print("i:", i, ", original:", raw[i], ", decrypted:", ref1[i])

        print("\nPart 2: FPGA encrypt, CPU decrypt")
        tmp_enc, _, tmp_exp = pi_c2p(encrypted)
        pen_recv = [PaillierEncryptedNumber(self._pub_key, tmp_enc[i], int(round(tmp_exp[i]))) for i in
                    range(TEST_SIZE)]
        ref2 = [self._priv_key.decrypt(v) for v in pen_recv]
        for i in range(TEST_SIZE):
            assert_diff(raw[i], ref2[i])
            print("i:", i, ", original:", raw[i], ", decrypted:", ref2[i])

        print("\nPart 3: CPU encrypt, FPGA decrypt")
        cpu_encrypted = [self._pub_key.encrypt(raw[i], None, bi_rand_vals[i]) for i in range(TEST_SIZE)]
        for i in range(TEST_SIZE):
            print("FPGA: i:", i, ", encoding:", pen_recv[i].ciphertext(False), ", exp:", pen_recv[i].exponent)
            print("CPU: i:", i, ", encoding:", cpu_encrypted[i].ciphertext(False), ", exp:", cpu_encrypted[i].exponent)
            assert pen_recv[i].ciphertext(False) == cpu_encrypted[i].ciphertext(False)
            assert pen_recv[i].exponent == cpu_encrypted[i].exponent
        pi_store = pi_p2c(None, cpu_encrypted, RAND_TYPE)
        ref3 = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, pi_store, None, None))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], ref3[i])
            print("i:", i, ", original:", raw[i], ", decrypted:", ref3[i])

        print("test passed")

    # tests both PEN and FPN transpose
    def test_transpose(self):
        print("\n\n", "*" * 100, "\n\nTest transpose of both FPN and PEN matrices Begins")
        raw = generate_rand(TEST_SIZE)
        # generate test PaillierEncryptedStorage and its shape
        te_store = te_p2c(raw, None)
        encoded = fp_encode(te_store, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded, None, None)
        rows, cols = 2, 3
        shape = TensorShapeStorage(rows, cols)
        pi_transpose_store, pi_transpose_shape = pi_transpose(encrypted, shape, None, None, None)
        fp_transpose_store, fp_transpose_shape = fp_transpose(encoded, shape, None, None, None)
        print("original shape:", shape.to_tuple(), ", transposed FPN shape:", fp_transpose_shape.to_tuple(),
              ", transposed PEN shape", pi_transpose_shape.to_tuple())
        assert pi_transpose_shape.to_tuple() == (cols, rows)
        assert fp_transpose_shape.to_tuple() == (cols, rows)
        fp_original = fp_c2p(encoded)
        fp_transposed = fp_c2p(fp_transpose_store)
        pi_original_cipher, pi_original_base, pi_original_exp = pi_c2p(encrypted)
        pi_transposed_cipher, pi_transposed_base, pi_transposed_exp = pi_c2p(pi_transpose_store)
        for i in range(rows):
            for j in range(cols):
                print("testing index (", i, ", ", j, ")")
                assert fp_original[i * cols + j].encoding == fp_transposed[j * rows + i].encoding
                assert fp_original[i * cols + j].BASE == fp_transposed[j * rows + i].BASE
                assert fp_original[i * cols + j].exponent == fp_transposed[j * rows + i].exponent
                assert pi_original_cipher[i * cols + j] == pi_transposed_cipher[j * rows + i]
                assert pi_original_base[i * cols + j] == pi_transposed_base[j * rows + i]
                assert pi_original_exp[i * cols + j] == pi_transposed_exp[j * rows + i]
        print("test passed")

    def test_pi_sum(self):
        print("\n\n", "*" * 100, "\n\nTest Sum Begins")
        # generate raw data
        raw = generate_rand(TEST_SIZE)
        # generate test PaillierEncryptedStorage and its shape
        te_store = te_p2c(raw, None)
        encoded = fp_encode(te_store, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded, None, None)
        rows, cols = 2, 3
        shape = TensorShapeStorage(rows, cols)

        print("raw matrix:\n", numpy.asarray(raw).reshape(rows, cols))

        print("TEST AXIS = 0")
        res_sum_axis0, res_shape_axis0 = pi_sum(self._fpga_pub_key, encrypted, shape, 0, None, None, None)
        res_axis0_fpga = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, res_sum_axis0, None, None))
        res_axis0_cpu = numpy.asarray(raw).reshape(rows, cols).sum(axis=0)
        print("result shape:", res_shape_axis0.to_tuple())
        for i in range(cols):
            print("column:", i, ", CPU result:", res_axis0_cpu[i], ", FPGA result:", res_axis0_fpga[i])
            assert_diff(res_axis0_cpu[i], res_axis0_fpga[i])

        print("TEST AXIS = 1")
        res_sum_axis1, res_shape_axis1 = pi_sum(self._fpga_pub_key, encrypted, shape, 1, None, None, None)
        res_axis1_fpga = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, res_sum_axis1, None, None))
        res_axis1_cpu = numpy.asarray(raw).reshape(rows, cols).sum(axis=1)
        print("result shape:", res_shape_axis1.to_tuple())
        for i in range(rows):
            print("column:", i, ", CPU result:", res_axis1_cpu[i], ", FPGA result:", res_axis1_fpga[i])
            assert_diff(res_axis1_cpu[i], res_axis1_fpga[i])

        print("TEST AXIS = None")
        res_sum_axis, res_shape_axis = pi_sum(self._fpga_pub_key, encrypted, shape, None, None, None, None)
        res_axis_fpga = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, res_sum_axis, None, None))
        res_axis_cpu = [numpy.asarray(raw).reshape(rows, cols).sum()]
        print("result shape:", res_shape_axis.to_tuple())
        for i in range(pow(CIPHER_BYTE, 0, PLAIN_BYTE)):
            print("result:", i, ", CPU result:", res_axis_cpu[i], ", FPGA result:", res_axis_fpga[i])
            assert_diff(res_axis_cpu[i], res_axis_fpga[i])
        print("test passed")

    def test_pi_matmul(self):
        print("\n\n", "*" * 100, "\n\nTest PEN Matrix_Multiplies FPN Begins")
        raw_1, raw_2 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        # generate the 2 operands
        te_store_1, te_store_2 = te_p2c(raw_1, None), te_p2c(raw_2, None)
        encoded_1, encoded_2 = fp_encode(te_store_1, self.n, self.max_int), fp_encode(te_store_2, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded_1, None, None)
        P, Q, R, S = 2, 3, 3, 2
        shape_1, shape_2 = TensorShapeStorage(P, Q), TensorShapeStorage(R, S)
        # then perform the matmul
        res_store, res_shape = pi_matmul(self._fpga_pub_key, encrypted, encoded_2, shape_1, shape_2, None, None, None)
        decrypted = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, res_store, None, None))
        res = numpy.asarray(decrypted).reshape(res_shape.to_tuple())
        ref = numpy.asarray(raw_1).reshape(P, Q) @ numpy.asarray(raw_2).reshape(R, S)
        print("FPGA result shape:", res_shape.to_tuple(), ", CPU result shape:", ref.shape)
        assert res_shape.to_tuple() == ref.shape
        print("CPU result:\n", ref, "\nFPGA result:\n", res)
        assert_ndarray_diff(res, ref)
        print("test passed")

    def test_combination(self):
        print("\n\n", "*" * 100, "\n\nTest Combination Begins")

        # generate operands
        raw_1, raw_3 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        if RAND_TYPE == INT64_TYPE:
            raw_2, raw_4 = [i % 16384 for i in generate_rand(TEST_SIZE)], [i % 16384 for i in generate_rand(TEST_SIZE)]
        elif RAND_TYPE == FLOAT_TYPE:
            raw_2, raw_4 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        else:
            raise PermissionError("Invalid Data Type")
        print('Raw data:\nraw_1:', raw_1, '\nraw_2:', raw_2, '\nraw_3:', raw_3, '\nraw_4:', raw_4)

        # generate shapes and NumPy arrays
        rows, cols = 2, 3
        array_1, array_2 = numpy.asarray(raw_1).reshape(rows, cols), numpy.asarray(raw_2).reshape(cols, rows)
        array_3, array_4 = numpy.asarray(raw_3).reshape(rows, cols), numpy.asarray(raw_4).reshape(rows, cols)
        shape_1, shape_2 = TensorShapeStorage(rows, cols), TensorShapeStorage(cols, rows)
        shape_3, shape_4 = TensorShapeStorage(rows, cols), TensorShapeStorage(rows, cols)

        # transfer and encode
        te_store_1, te_store_2 = te_p2c(raw_1, None), te_p2c(raw_2, None)
        te_store_3, te_store_4 = te_p2c(raw_3, None), te_p2c(raw_4, None)
        encoded_1, encoded_2 = fp_encode(te_store_1, self.n, self.max_int), fp_encode(te_store_2, self.n, self.max_int)
        encoded_3, encoded_4 = fp_encode(te_store_3, self.n, self.max_int), fp_encode(te_store_4, self.n, self.max_int)

        # perform encrypt and obfs
        encrypted_old_1 = pi_encrypt(self._fpga_pub_key, encoded_1, None, None)
        encrypted_old_3 = pi_encrypt(self._fpga_pub_key, encoded_3, None, None)
        rand_store_1 = pi_gen_obf_seed(None, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, time.time(), None)
        rand_store_3 = pi_gen_obf_seed(None, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, time.time(), None)
        encrypted_1 = pi_obfuscate(self._fpga_pub_key, encrypted_old_1, rand_store_1, None, None)
        encrypted_3 = pi_obfuscate(self._fpga_pub_key, encrypted_old_3, rand_store_3, None, None)

        print("Perform Add")
        add_res_store, add_res_shape = pi_add(self._fpga_pub_key, encrypted_1, encrypted_3, shape_1, shape_3, None,
                                              None, None)
        add_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, add_res_store, None, None)).reshape(
            add_res_shape.to_tuple())
        add_ref = array_1 + array_3
        print("FPGA intermediate result:", add_res)
        print("NumPy intermediate result:", add_ref)
        assert_ndarray_diff(add_res, add_ref)

        print("Perform Mul")
        mul_res_store, mul_res_shape = pi_mul(self._fpga_pub_key, add_res_store, encoded_4, add_res_shape, shape_4,
                                              None, None, None)
        mul_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, mul_res_store, None, None)).reshape(
            mul_res_shape.to_tuple())
        mul_ref = (array_1 + array_3) * array_4
        print("FPGA intermediate result:", mul_res)
        print("NumPy intermediate result:", mul_ref)
        assert_ndarray_diff(mul_res, mul_ref)

        print("Perform Matmul")
        matmul_res_store, matmul_res_shape = pi_matmul(self._fpga_pub_key, mul_res_store, encoded_2, mul_res_shape,
                                                       shape_2, None, None, None)
        matmul_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, matmul_res_store, None, None)).reshape(
            matmul_res_shape.to_tuple())
        matmul_ref = ((array_1 + array_3) * array_4) @ array_2
        print("FPGA result shape:", matmul_res_shape.to_tuple(), ", CPU result shape:", matmul_ref.shape)
        print("CPU result:\n", matmul_ref)
        print("FPGA result:\n", matmul_res)
        assert_ndarray_diff(matmul_res, matmul_ref)

        print("test passed")

    def test_c2bytes_and_bytes2c(self):
        print("\n\n", "*" * 100, "\n\nTest bytes and c transformation begins")

        raw = generate_rand(TEST_SIZE)
        print("Raw Data:", raw)

        print("\nPart 1: test te_c2bytes and te_bytes2c")
        te_store = te_p2c(raw, None)
        te_bytes = te_c2bytes(te_store, None)
        te_store_recv = te_bytes2c(te_bytes, te_store)
        te_ref = list(te_c2p(te_store_recv))
        print("Bytes Representation:", te_bytes)
        print("Received data:", te_ref)
        assert te_store.data_type == RAND_TYPE
        assert te_store_recv.data_type == RAND_TYPE
        for i in range(TEST_SIZE):
            assert_diff(raw[i], te_ref[i])

        print("\nPart 2: test fp_c2bytes and fp_bytes2c")
        fp_store = fp_encode(te_store, self.n, self.max_int)
        fp_bytes = fp_c2bytes(fp_store, None)
        fp_store_recv = fp_bytes2c(fp_bytes, fp_store)
        fp_ref = list(te_c2p(fp_decode(fp_store_recv, None, None)))
        print("Bytes Representation (excerpt):", fp_bytes[1888:1999])
        print("Received data:", fp_ref)
        assert fp_store.data_type == RAND_TYPE
        assert fp_store_recv.data_type == RAND_TYPE
        for i in range(TEST_SIZE):
            assert_diff(raw[i], fp_ref[i])

        print("\nPart 3: test pi_c2bytes and pi_bytes2c")
        pi_store = pi_encrypt(self._fpga_pub_key, fp_store, None, None)
        pi_bytes = pi_c2bytes(pi_store, None)
        pi_store_recv = pi_bytes2c(pi_bytes, pi_store)
        pi_ref = list(te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, pi_store_recv, None, None)))
        print("Bytes Representation (excerpt):", pi_bytes[1888:1999])
        print("Received data:", pi_ref)
        assert pi_store.data_type == RAND_TYPE
        assert pi_store_recv.data_type == RAND_TYPE
        for i in range(TEST_SIZE):
            assert_diff(raw[i], pi_ref[i])

        print("test passed")

    def test_fp_slice(self):
        print("\n\n", "*" * 100, "\n\nTest fp_slice begins")
        rows, cols = 3, 4
        shape = [rows, cols]
        begin_h, end_h = 2, 3
        begin_v, end_v = 1, 3
        raw = numpy.asarray(generate_rand(functools.reduce(operator.mul, [*shape], 1))).reshape(shape)
        store = te_p2c(raw, None)
        encoded = fp_encode(store, self.n, self.max_int)
        slice_h_store, slice_h_shape = fp_slice(encoded, TensorShapeStorage(*shape), begin_h, end_h, 0, None, None,
                                                None)
        slice_v_store, slice_v_shape = fp_slice(encoded, TensorShapeStorage(*shape), begin_v, end_v, 1, None, None,
                                                None)
        recv_h = numpy.asarray(te_c2p(fp_decode(slice_h_store, None, None))).reshape(slice_h_shape)
        recv_v = numpy.asarray(te_c2p(fp_decode(slice_v_store, None, None))).reshape(slice_v_shape)
        print("raw array:\n", raw)
        print("horizontal slice:\n", recv_h)
        print("vertical slice:\n", recv_v)
        for i in range(end_h - begin_h):
            for j in range(cols):
                assert_diff(raw[begin_h + i][j], recv_h[i][j])
        for i in range(rows):
            for j in range(end_v - begin_v):
                assert_diff(raw[i][begin_v + j], recv_v[i][j])
        assert slice_h_store.data_type == RAND_TYPE
        assert slice_v_store.data_type == RAND_TYPE
        print("test passed")

    def test_pi_slice(self):
        print("\n\n", "*" * 100, "\n\nTest pi_slice begins")
        rows, cols = 3, 4
        shape = [rows, cols]
        begin_h, end_h = 2, 3
        begin_v, end_v = 1, 3
        raw = numpy.asarray(generate_rand(functools.reduce(operator.mul, [*shape], 1))).reshape(shape)
        store = te_p2c(raw, None)
        encoded = fp_encode(store, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded, None, None)
        slice_h_store, slice_h_shape = pi_slice(encrypted, TensorShapeStorage(*shape), begin_h, end_h, 0, None, None,
                                                None)
        slice_v_store, slice_v_shape = pi_slice(encrypted, TensorShapeStorage(*shape), begin_v, end_v, 1, None, None,
                                                None)
        recv_h = numpy.asarray(
            te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, slice_h_store, None, None))).reshape(
            slice_h_shape)
        recv_v = numpy.asarray(
            te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, slice_v_store, None, None))).reshape(
            slice_v_shape)
        print("raw array:\n", raw)
        print("horizontal slice:\n", recv_h)
        print("vertical slice:\n", recv_v)
        for i in range(end_h - begin_h):
            for j in range(cols):
                assert_diff(raw[begin_h + i][j], recv_h[i][j])
        for i in range(rows):
            for j in range(end_v - begin_v):
                assert_diff(raw[i][begin_v + j], recv_v[i][j])
        assert slice_h_store.data_type == RAND_TYPE
        assert slice_v_store.data_type == RAND_TYPE
        print("test passed")

    def test_pi_reshape(self):
        print("\n\n", "*" * 100, "\n\nTest pi_reshape begins")
        raw = generate_rand(TEST_SIZE)
        store = te_p2c(raw, None)
        encoded = fp_encode(store, self.n, self.max_int)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded, None, None)
        old_shape, new_shape = TensorShapeStorage(2, 3), TensorShapeStorage(3, 2)
        new_store_res, new_shape_res = pi_reshape(encrypted, old_shape, new_shape, encrypted, None,
                                                  None)  # PREVENT DOUBLE FREE: option 1
        print("PyObject ids before and after reshape:", id(new_store_res), id(encrypted))
        assert id(new_store_res) == id(encrypted)
        # encrypted.exp_storage, encrypted.pen_storage, encrypted.base_storage =
        # None, None, None  # PREVENT DOUBLE FREE: option 2

        print("original shape:", old_shape.to_tuple(), ", returned shape:", new_shape_res.to_tuple(),
              ", expected new shape:", new_shape.to_tuple())
        assert new_shape.to_tuple() == new_shape_res.to_tuple()
        recv = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, new_store_res, None, None))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], recv[i])
        assert encoded.data_type == RAND_TYPE
        assert encrypted.data_type == RAND_TYPE
        assert new_store_res.data_type == RAND_TYPE
        print("raw tensor:\n", numpy.asarray(raw).reshape(old_shape.to_tuple()))
        print("reshaped tensor:\n", numpy.asarray(recv).reshape(new_shape_res.to_tuple()))

        print("test passed")

    def test_fp_mul(self):
        print("\n\n", "*" * 100, "\n\nTest fp_mul begins")
        raw_1, raw_2 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        store_1, store_2 = te_p2c(raw_1, None), te_p2c(raw_2, None)
        encoded_1, encoded_2 = fp_encode(store_1, self.n, self.max_int), fp_encode(store_2, self.n, self.max_int)
        res_store, res_shape = fp_mul(encoded_1, encoded_2, TensorShapeStorage(2, 3), TensorShapeStorage(2, 3), None,
                                      None, None)
        decoded = fp_decode(res_store, None, None)
        recv = te_c2p(decoded)
        assert res_shape.to_tuple() == (2, 3)
        assert encoded_1.data_type == RAND_TYPE
        assert encoded_2.data_type == RAND_TYPE
        assert res_store.data_type == RAND_TYPE
        assert decoded.data_type == RAND_TYPE

        cpu_encoded_1 = [FixedPointNumber.encode(v, self.n, self.max_int) for v in raw_1]
        cpu_encoded_2 = [FixedPointNumber.encode(v, self.n, self.max_int) for v in raw_2]
        cpu_res = [FixedPointNumber((cpu_encoded_1[i].encoding * cpu_encoded_2[i].encoding) % self.n,
                                    cpu_encoded_1[i].exponent + cpu_encoded_2[i].exponent, self.n, self.max_int) for i
                   in range(TEST_SIZE)]
        cpu_ref = [v.decode() for v in cpu_res]

        print("FPGA result:", list(recv))
        print("CPU result:", list(cpu_ref))

        res_fp = fp_c2p(res_store)
        for i in range(TEST_SIZE):
            assert_diff(recv[i], cpu_ref[i])
            assert_diff(res_fp[i].encoding, cpu_res[i].encoding)
            assert res_fp[i].BASE == cpu_res[i].BASE
            assert res_fp[i].exponent == cpu_res[i].exponent

        print("test passed")

    def test_te_c2p_first(self):
        print("\n\n", "*" * 100, "\n\nTest te_c2p_first begins")

        raw = generate_rand(TEST_SIZE)
        store = te_p2c(raw, None)
        print(raw[0], te_c2p_first(store))
        assert raw[0] == te_c2p_first(store)

        print("test passed")

    def test_malloc(self):
        print("\n\n", "*" * 100, "\n\nTest malloc begins")

        bi_store = bi_alloc(None, TEST_SIZE, PLAIN_BYTE, MEM_HOST)
        te_store = te_alloc(None, TEST_SIZE, MEM_HOST)
        fp_store = fp_alloc(None, TEST_SIZE, MEM_HOST)
        pi_store = pi_alloc(None, TEST_SIZE, MEM_HOST)

        raw = generate_rand(TEST_SIZE)
        store = te_p2c(raw, te_store)
        encoded = fp_encode(store, self.n, self.max_int, None, None, fp_store, None)
        print("PyObject ids before and after encode:", id(encoded), id(fp_store))
        assert id(encoded) == id(fp_store)
        encrypted = pi_encrypt(self._fpga_pub_key, encoded, pi_store, None)
        obf_seeds = pi_gen_obf_seed(bi_store, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, time.time(), None)
        encrypted = pi_obfuscate(self._fpga_pub_key, pi_store, obf_seeds, pi_store, None)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, encrypted, te_store, None)
        received = te_c2p(decrypted)
        print("raw data:", raw, "\nreceived data:", list(received))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], received[i])

        bi_free(bi_store)
        te_free(te_store)
        fp_free(fp_store)
        pi_free(pi_store)
        # fp_store.base_storage, fp_store.bigint_storage, fp_store.exp_storage = None, None, None

        print("test passed")

    def test_p2c(self):
        print("\n\n", "*" * 100, "\n\nTest fp_p2c & pi_p2c Begins")

        print("Part 1.1: test te_p2c for list")
        raw = generate_rand(TEST_SIZE)
        te_store = te_p2c(raw, None)
        received = te_c2p(te_store)
        print("raw data:", raw, "\nreceived data:", list(received))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], received[i])

        print("Part 1.2: test te_p2c for ndarray")
        np_raw = numpy.asarray(raw).reshape(2, 3)
        te_store = te_p2c(np_raw, None)
        received = te_c2p(te_store)
        print("raw data:", raw, "\nreceived data:", list(received))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], received[i])

        print("Part 2.1: test fp_p2c for list")
        cpu_encoded = [FixedPointNumber.encode(v, self.n, self.max_int) for v in raw]
        fp_store = fp_p2c(None, cpu_encoded, RAND_TYPE)
        decoded = te_c2p(fp_decode(fp_store, None, None))
        print("raw data:", raw, "\ndecoded data:", list(decoded))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], decoded[i])

        print("Part 2.2: test fp_p2c for ndarray")
        np_cpu_encoded = numpy.asarray(cpu_encoded).reshape(2, 3)
        fp_store = fp_p2c(None, np_cpu_encoded, RAND_TYPE)
        decoded = te_c2p(fp_decode(fp_store, None, None))
        print("raw data:", raw, "\ndecoded data:", list(decoded))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], decoded[i])

        print("Part 3.1: test pi_p2c for list")
        cpu_encrypted = [self._pub_key.encrypt(raw[i], None, 0) for i in range(TEST_SIZE)]
        pi_store = pi_p2c(None, cpu_encrypted, RAND_TYPE)
        decrypted = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, pi_store, None, None))
        print("raw data:", raw, "\ndecrypted data:", list(decrypted))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], decrypted[i])

        print("Part 3.2: test pi_p2c for ndarray")
        np_cpu_encrypted = numpy.asarray(cpu_encrypted).reshape(2, 3)
        pi_store = pi_p2c(None, np_cpu_encrypted, RAND_TYPE)
        decrypted = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, pi_store, None, None))
        print("raw data:", raw, "\ndecrypted data:", list(decrypted))
        for i in range(TEST_SIZE):
            assert_diff(raw[i], decrypted[i])

        print("test passed")

    def test_tensor(self):
        print("\n\n", "*" * 100, "\n\nTest Tensor begins")
        if RAND_TYPE == INT64_TYPE:
            raw_1, raw_2 = [i % 128 + 1 for i in generate_rand(TEST_SIZE)], [i % 128 + 1 for i in
                                                                             generate_rand(TEST_SIZE)]
        elif RAND_TYPE == FLOAT_TYPE:
            raw_1, raw_2 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        else:
            raise PermissionError("Invalid Data Type")
        rows, cols = 2, 3
        shape = TensorShapeStorage(rows, cols)
        transposed_shape = TensorShapeStorage(cols, rows)
        array_1 = numpy.asarray(raw_1).reshape(shape.to_tuple())
        array_2 = numpy.asarray(raw_2).reshape(shape.to_tuple())
        array_3 = array_2.transpose()
        store_1 = TensorStorage(array_1, TEST_SIZE, MEM_HOST, RAND_TYPE)
        store_2 = TensorStorage(array_2, TEST_SIZE, MEM_HOST, RAND_TYPE)
        store_3 = TensorStorage(array_3, TEST_SIZE, MEM_HOST, RAND_TYPE)
        print("raw data:\n", array_1, "\n", array_2)

        print("Part 1: test shape")

        def __run_test_shape(dims):
            shape = tuple(dims)
            c_shape = te_p2c_shape(shape, None)
            py_shape = te_c2p_shape(c_shape)
            print("compare shapes:", shape, c_shape.to_tuple(), py_shape)
            assert shape == c_shape.to_tuple()
            assert shape == py_shape

        __run_test_shape([])
        __run_test_shape([1])
        __run_test_shape([1, 2])

        print("Part 2: test te_slice")
        res_store, res_shape = te_slice(store_1, shape, 1, 2, 0, None, None, None)
        assert (res_store.data == array_1[1:2]).all()
        assert res_shape.to_tuple() == (1, cols)
        res_store, res_shape = te_slice(store_1, shape, 0, 2, 1, None, None, None)
        assert (res_store.data == array_1[:, 0:2]).all()
        assert res_shape.to_tuple() == (rows, 2)

        print("Part 3: test te_cat")
        res_store, res_shape = te_cat([store_1, store_2], 0, None, None)
        assert (res_store.data == numpy.vstack((array_1, array_2))).all()
        assert res_shape.to_tuple() == (rows * 2, cols)
        res_store, res_shape = te_cat([store_1, store_2], 1, None, None)
        assert (res_store.data == numpy.hstack((array_1, array_2))).all()
        assert res_shape.to_tuple() == (rows, cols * 2)

        print("Part 4: test te_pow")
        res_store, res_shape = te_pow(store_1, 9, shape, None, None, None)
        assert (res_store.data == array_1 ** 9).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 5: test te_add")
        res_store, res_shape = te_add(store_1, store_2, shape, shape, None, None, None)
        assert (res_store.data == array_1 + array_2).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 6: test te_mul")
        res_store, res_shape = te_mul(store_1, store_2, shape, shape, None, None, None)
        assert (res_store.data == array_1 * array_2).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 7: test te_truediv")
        res_store, res_shape = te_truediv(store_1, store_2, shape, shape, None, None, None)
        assert (res_store.data == array_1 / array_2).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 8: test te_floordiv")
        res_store, res_shape = te_floordiv(store_1, store_2, shape, shape, None, None, None)
        assert (res_store.data == array_1 // array_2).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 9: test te_sub")
        res_store, res_shape = te_sub(store_1, store_2, shape, shape, None, None, None)
        assert (res_store.data == array_1 - array_2).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 10: test te_matmul")
        res_store, res_shape = te_matmul(store_1, store_3, shape, transposed_shape, None, None, None)
        print(res_store.data)
        assert_ndarray_diff(res_store.data, array_1 @ array_2.transpose())
        assert res_shape.to_tuple() == (rows, rows)

        print("Part 11: test te_abs")
        res_store, res_shape = te_abs(store_1, shape, None, None, None)
        assert (res_store.data == abs(array_1)).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 12: test te_neg")
        res_store, res_shape = te_neg(store_1, shape, None, None, None)
        assert (res_store.data == -array_1).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 13: test te_transpose")
        res_store, res_shape = te_transpose(store_1, shape, None, None, None)
        assert (res_store.data == array_1.transpose()).all()
        assert res_shape.to_tuple() == transposed_shape.to_tuple()

        print("Part 14: test te_sum")
        res_store, res_shape = te_sum(store_1, shape, None, None, None, None)
        assert (res_store.data == array_1.sum()).all()
        assert res_shape.to_tuple() == ()
        res_store, res_shape = te_sum(store_1, shape, 0, None, None, None)
        assert (res_store.data == array_1.sum(axis=0)).all()
        assert res_shape.to_tuple() == (cols,)
        res_store, res_shape = te_sum(store_1, shape, 1, None, None, None)
        assert (res_store.data == array_1.sum(axis=1)).all()
        assert res_shape.to_tuple() == (rows,)

        print("Part 15: test te_reshape")
        res_store, res_shape = te_reshape(store_1, shape, transposed_shape, None, None, None)
        assert (res_store.data == array_1.reshape(transposed_shape.to_tuple())).all()
        assert res_shape.to_tuple() == transposed_shape.to_tuple()

        print("Part 16: test te_exp")
        res_store, res_shape = te_exp(store_1, shape, None, None, None)
        assert (res_store.data == numpy.exp(array_1)).all()
        assert res_shape.to_tuple() == shape.to_tuple()

        print("Part 17: test te_hstack")
        res_store, res_shape = te_hstack(store_1, store_2, shape, shape, None, None, None)
        assert (res_store.data == numpy.hstack((array_1, array_2))).all()
        assert res_shape.to_tuple() == (rows, cols * 2)

        print("Test passed")

    def test_matmul_fix(self):
        print("\n\n", "*" * 100, "\n\nTest matmul_fix Begins")
        print("This test is to test whether the previous overflow bug in matmul has been fixed")

        # use specific operands
        raw_1 = [-6.328172916615867, -2.8424299647675904, 5.161324580891171, -0.23598534366587853, 0.8092957262188305,
                 19.50497470592641]
        raw_2 = [-0.048743928478232584, 6.191889562038381, 2.7177577835259017, 17.09697900858307, 11.31935499510339,
                 -4.881758293445916]
        raw_3 = [14.051643909583548, 5.246105161671397, 6.764067053406746, 4.727717881071932, -6.361020843266641,
                 -12.94175161066905]
        raw_4 = [-0.003912522017777569, 14.519125724575714, -5.401608455748054, 13.918193685722846, 5.97460357170185,
                 -3.960383753671568]

        print('Raw data:\n', raw_1, '\n', raw_2, '\n', raw_3, '\n', raw_4)

        # generate shapes and NumPy arrays
        rows, cols = 2, 3
        array_1, array_2 = numpy.asarray(raw_1).reshape(rows, cols), numpy.asarray(raw_2).reshape(cols, rows)
        array_3, array_4 = numpy.asarray(raw_3).reshape(rows, cols), numpy.asarray(raw_4).reshape(rows, cols)
        shape_1, shape_2 = TensorShapeStorage(rows, cols), TensorShapeStorage(cols, rows)
        shape_3, shape_4 = TensorShapeStorage(rows, cols), TensorShapeStorage(rows, cols)

        # transfer and encode
        te_store_1, te_store_2 = te_p2c(raw_1, None), te_p2c(raw_2, None)
        te_store_3, te_store_4 = te_p2c(raw_3, None), te_p2c(raw_4, None)
        encoded_1, encoded_2 = fp_encode(te_store_1, self.n, self.max_int), fp_encode(te_store_2, self.n, self.max_int)
        encoded_3, encoded_4 = fp_encode(te_store_3, self.n, self.max_int), fp_encode(te_store_4, self.n, self.max_int)

        # perform encrypt and obfs
        encrypted_old_1 = pi_encrypt(self._fpga_pub_key, encoded_1, None, None)
        encrypted_old_3 = pi_encrypt(self._fpga_pub_key, encoded_3, None, None)
        rand_store_1 = pi_gen_obf_seed(None, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, time.time(), None)
        rand_store_3 = pi_gen_obf_seed(None, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, time.time(), None)
        encrypted_1 = pi_obfuscate(self._fpga_pub_key, encrypted_old_1, rand_store_1, None, None)
        encrypted_3 = pi_obfuscate(self._fpga_pub_key, encrypted_old_3, rand_store_3, None, None)

        print("Perform Add")
        add_res_store, add_res_shape = pi_add(self._fpga_pub_key, encrypted_1, encrypted_3, shape_1, shape_3, None,
                                              None, None)
        add_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, add_res_store, None, None)).reshape(
            add_res_shape.to_tuple())
        add_ref = array_1 + array_3
        print("FPGA intermediate result:", add_res)
        print("NumPy intermediate result:", add_ref)
        assert_ndarray_diff(add_res, add_ref)

        print("Perform Mul")
        mul_res_store, mul_res_shape = pi_mul(self._fpga_pub_key, add_res_store, encoded_4, add_res_shape, shape_4,
                                              None, None, None)
        mul_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, mul_res_store, None, None)).reshape(
            mul_res_shape.to_tuple())
        mul_ref = (array_1 + array_3) * array_4
        print("FPGA intermediate result:", mul_res)
        print("NumPy intermediate result:", mul_ref)
        assert_ndarray_diff(mul_res, mul_ref)

        # The following code is to dump PEN and FPN storages into stdout
        # print("n (big endian bytes):", self._pub_key.n.to_bytes(CIPHER_BYTE, 'big').hex())
        # print("nsquare (big endian bytes):", self._pub_key.nsquare.to_bytes(CIPHER_BYTE, 'big').hex())
        # fp_list = fp_c2p(encoded_2)
        # pi_cipher, pi_base, pi_exp = pi_c2p(mul_res_store)
        # print("\n\n>>>>>>>>>>>>>> dumping pen storage\n")
        # for i in range(TEST_SIZE):
        #     print("=====================id:", i)
        #     print("PEN cipher (big endian bytes):", pi_cipher[i].to_bytes(CIPHER_BYTE, 'big').hex())
        #     print("PEN base (decimal):", pi_base[i])
        #     print("PEN exponent (decimal):", pi_exp[i])
        # print("\n\n>>>>>>>>>>>>>> dumping fpn storage\n")
        # for i in range(TEST_SIZE):
        #     print("=====================id:", i)
        #     print("FPN encoding (big endian bytes):", fp_list[i].encoding.to_bytes(CIPHER_BYTE, 'big').hex())
        #     print("FPN base (decimal):", fp_list[i].BASE)
        #     print("FPN exponent (decimal):", fp_list[i].exponent)

        # The following code is essentially to decrypt and encrypt again.
        # However, the numbers might be truncated so that the overflow could be mitigated
        # tmp_te_store = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, mul_res_store, None, None)
        # mul_res_store = pi_encrypt(self._fpga_pub_key, fp_encode(tmp_te_store, self.n, self.max_int), None, None)
        # mul_res_store = pi_obfuscate(self._fpga_pub_key, mul_res_store, rand_store_1, None, None)

        print("Perform Matmul: PEN shape (2, 3), FPN shape (3, 2)")
        matmul_res_store, matmul_res_shape = pi_matmul(self._fpga_pub_key, mul_res_store, encoded_2, mul_res_shape,
                                                       shape_2, None, None, None)
        matmul_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, matmul_res_store, None, None)).reshape(
            matmul_res_shape.to_tuple())
        matmul_ref = ((array_1 + array_3) * array_4) @ array_2
        print("FPGA result shape:", matmul_res_shape.to_tuple(), ", CPU result shape:", matmul_ref.shape)
        print("CPU result:\n", matmul_ref)
        print("FPGA result:\n", matmul_res)
        assert_ndarray_diff(matmul_res, matmul_ref)

        print("test passed")

    def test_te_vertical_slice(self):
        print("\n\n", "*" * 100, "\n\nTest Tensor Vertical Slice Begins")
        shape = (2, 3)
        np_raw = numpy.asarray(generate_rand(TEST_SIZE)).reshape(shape)
        print("raw data:\n", np_raw)
        np_raw_store = TensorStorage(np_raw, TEST_SIZE, MEM_HOST, RAND_TYPE)
        np_slice_store, np_slice_shape = te_slice(np_raw_store, TensorShapeStorage(*shape), 2, 3, 1, None, None, None)
        print("numpy slice data:\n", np_slice_store.data)
        c_slice_store = te_p2c(np_slice_store.data, None)
        slice_recv = te_c2p(c_slice_store).reshape(np_slice_shape)
        print("received slice data:\n", slice_recv)
        assert_ndarray_diff(np_slice_store.data, slice_recv)
        print("Test Passed")

    def test_encode_precision_1(self):
        print("\n\n", "*" * 100, "\n\nTesting encode with precision 1")
        raw = [19.12634]
        store = te_p2c(raw, None)
        fp_store = fp_encode(store, self.n, self.max_int, 1)
        recv = fp_decode(fp_store, None, None)
        recv_scalar = te_c2p(recv)
        print("result:", recv_scalar)
        assert recv_scalar[0] == 19
        print("Test passed")

    def test_matmul_limits(self):
        print("\n\n", "*" * 100, "\n\nTest after how many matmul would cause our internal data structure overflow")
        shape_tuple = (TEST_SIZE // 2, TEST_SIZE // 2)
        shape_store = TensorShapeStorage(*shape_tuple)
        shape_size = functools.reduce(operator.mul, [*shape_tuple], 1)
        raw_1, raw_2 = [random.gauss(0, 1) for _ in range(shape_size)], [random.gauss(0, 1) for _ in range(shape_size)]
        left_array, right_array = numpy.asarray(raw_1).reshape(shape_tuple), numpy.asarray(raw_2).reshape(shape_tuple)

        # FPGA encode & encrypt
        left_store = pi_encrypt(self._fpga_pub_key, fp_encode(te_p2c(raw_1, None), self.n, self.max_int), None, None)
        obf_seeds = pi_gen_obf_seed(None, self._fpga_pub_key, shape_size, CIPHER_BITS // 6, time.time(), None)
        left_store = pi_obfuscate(self._fpga_pub_key, left_store, obf_seeds, left_store, None)
        right_store = fp_encode(te_p2c(raw_2, None), self.n, self.max_int)

        for i in range(1, 100):
            # Dumping useful data
            print("\n>>>>>>>>>>>>>>> iteration:", i)
            _, base, exp = pi_c2p(left_store)
            fp_py_store = fp_c2p(right_store)
            all_exponents = [*exp, *[v.exponent for v in fp_py_store]]
            max_exponent = max(*all_exponents)
            if i == 1:
                initial_max_exp = max_exponent
            print("all exponents:", all_exponents)
            print("max base:", max(*base, *[v.BASE for v in fp_py_store]), ", max exponent:", max_exponent)

            # Running Numpy and FPGA matmul, storing the result to the left operand
            left_array = left_array @ right_array
            left_store, tmp_shape = pi_matmul(self._fpga_pub_key, left_store, right_store, shape_store, shape_store,
                                              left_store, None, None)

            # Get matmul result of the current iteration and compare
            tmp_res = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, left_store, None, None)).reshape(
                tmp_shape.to_tuple())
            print("FPGA result:\n", tmp_res, "\nCPU result:\n", left_array)
            try:
                assert_ndarray_diff(tmp_res, left_array)
            except AssertionError:
                final_exponents = [*pi_c2p(left_store)[2], *[v.exponent for v in fp_c2p(right_store)]]
                final_max_exp = max(*final_exponents)
                print("final exponents:", final_exponents)
                print("initial max exponent:", initial_max_exp, ", final max exponent", final_max_exp)
                print(">>>>> FPGA and CPU results didn't match at iteration #{}.".format(i))
                # The following assertions are deprecated as we treat max_exp for FPN and PEN separately
                # assert 256 <= final_max_exp < 512
                # assert initial_max_exp * int(round(2 ** i)) == final_max_exp
                # assert i == math.ceil(8 - math.log2(initial_max_exp))
                break

        print("Test passed")

    def test_fp_cat(self):
        print("\n\n", "*" * 100, "\n\nTest fp_cat begins")
        shape_tuple = (2, 3)
        shape = TensorShapeStorage(*shape_tuple)
        shape_size = int(round(numpy.prod(shape_tuple)))
        tmp_1, tmp_2 = generate_rand(shape_size), generate_rand(shape_size)
        array_1, array_2 = numpy.asarray(tmp_1).reshape(shape_tuple), numpy.asarray(tmp_2).reshape(shape_tuple)
        print("array_1:\n", array_1, "\narray_2:\n", array_2)

        fp_store_1 = fp_encode(te_p2c(array_1, None), self.n, self.max_int)
        fp_store_2 = fp_encode(te_p2c(array_2, None), self.n, self.max_int)

        # test vertical cat
        print("Part 1: test vertical cat")
        cat_store, cat_shape = fp_cat([fp_store_1, fp_store_2], [shape, shape], 0, None, None)
        print("result shape:", cat_shape.to_tuple())
        assert cat_shape.to_tuple() == (shape_tuple[0] * 2, shape_tuple[1])
        decoded = fp_decode(cat_store, None, None)
        res = te_c2p(decoded).reshape(cat_shape.to_tuple())
        ref = numpy.concatenate((array_1, array_2), 0)
        print("result tensor:\n", res)
        print("reference tensor:\n", ref)
        assert_ndarray_diff(res, ref)

        # test horizontal cat
        print("Part 2: test horizontal cat")
        cat_store, cat_shape = fp_cat([fp_store_1, fp_store_2], [shape, shape], 1, None, None)
        print("result shape:", cat_shape.to_tuple())
        assert cat_shape.to_tuple() == (shape_tuple[0], shape_tuple[1] * 2)
        decoded = fp_decode(cat_store, None, None)
        res = te_c2p(decoded).reshape(cat_shape.to_tuple())
        ref = numpy.concatenate((array_1, array_2), 1)
        print("result tensor:\n", res)
        print("reference tensor:\n", ref)
        assert_ndarray_diff(res, ref)

        print("test passed")

    def test_pi_cat(self):
        print("\n\n", "*" * 100, "\n\nTest pi_cat begins")
        shape_tuple = (2, 3)
        shape = TensorShapeStorage(*shape_tuple)
        shape_size = int(round(numpy.prod(shape_tuple)))
        array_1 = numpy.asarray(generate_rand(shape_size)).reshape(shape_tuple)
        array_2 = numpy.asarray(generate_rand(shape_size)).reshape(shape_tuple)
        array_3 = numpy.asarray(generate_rand(shape_size)).reshape(shape_tuple)
        print("array_1:\n", array_1, "\narray_2:\n", array_2, "\narray_3:\n", array_3)

        fp_store_1 = fp_encode(te_p2c(array_1, None), self.n, self.max_int)
        fp_store_2 = fp_encode(te_p2c(array_2, None), self.n, self.max_int)
        fp_store_3 = fp_encode(te_p2c(array_3, None), self.n, self.max_int)
        pi_store_1 = pi_encrypt(self._fpga_pub_key, fp_store_1, None, None)
        pi_store_2 = pi_encrypt(self._fpga_pub_key, fp_store_2, None, None)
        pi_store_3 = pi_encrypt(self._fpga_pub_key, fp_store_3, None, None)

        # test horizontal cat
        print("Part 1: test horizontal cat")
        cat_store, cat_shape = pi_cat([pi_store_1, pi_store_2, pi_store_3], [shape, shape, shape], 1, None, None)
        print("result shape:", cat_shape.to_tuple())
        assert cat_shape.to_tuple() == (shape_tuple[0], shape_tuple[1] * 3)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, cat_store, None, None)
        res = te_c2p(decrypted).reshape(cat_shape.to_tuple())
        ref = numpy.concatenate((array_1, array_2, array_3), 1)
        print("result tensor:\n", res)
        print("reference tensor:\n", ref)
        assert_ndarray_diff(res, ref)

        # test vertical cat
        print("Part 2: test vertical cat")
        cat_store, cat_shape = pi_cat([pi_store_1, pi_store_2, pi_store_3], [shape, shape, shape], 0, None, None)
        print("result shape:", cat_shape.to_tuple())
        assert cat_shape.to_tuple() == (shape_tuple[0] * 3, shape_tuple[1])
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, cat_store, None, None)
        res = te_c2p(decrypted).reshape(cat_shape.to_tuple())
        ref = numpy.concatenate((array_1, array_2, array_3), 0)
        print("result tensor:\n", res)
        print("reference tensor:\n", ref)
        assert_ndarray_diff(res, ref)

        # test concat combination
        print("Part 3: test combined cat")
        vector_size = cat_shape.to_tuple()[0]
        vector_shape = TensorShapeStorage(vector_size, 1)
        vector = numpy.asarray(generate_rand(vector_size)).reshape(vector_size, 1)
        print("vector:\n", vector)
        vector_te_store = te_p2c(vector, None)
        vector_fp_store = fp_encode(vector_te_store, self.n, self.max_int)
        vector_pi_store = pi_encrypt(self._fpga_pub_key, vector_fp_store, None, None)
        cat_store, cat_shape = pi_cat([cat_store, vector_pi_store], [cat_shape, vector_shape], 1, None, None)
        print("result shape:", cat_shape.to_tuple())
        assert cat_shape.to_tuple() == (shape_tuple[0] * 3, shape_tuple[1] + 1)
        decrypted = pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, cat_store, None, None)
        res = te_c2p(decrypted).reshape(cat_shape.to_tuple())
        ref = numpy.concatenate([numpy.concatenate((array_1, array_2, array_3), 0), vector], 1)
        print("result tensor:\n", res)
        print("reference tensor:\n", ref)
        assert_ndarray_diff(res, ref)

        print("test passed")


if __name__ == "__main__":
    unittest.main()
