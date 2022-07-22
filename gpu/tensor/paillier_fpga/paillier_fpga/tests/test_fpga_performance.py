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
import time

from fate_arch.tensor.impl.blocks.python_paillier_block import (
    PaillierKeypair,
    PaillierEncryptedNumber,
    FixedPointNumber,
    gmpy_math,
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
    pi_free, initialize_device, pi_matmul,
)

RAND_TYPE = FLOAT_TYPE  # SWITCH DATA TYPE HERE: EITHER INT64_TYPE OR FLOAT_TYPE
NUM_ROWS = 666
NUM_COLS = 666
TEST_SIZE = NUM_ROWS * NUM_COLS
ERROR_TOLERANCE = 1e-10


def generate_rand(test_size):
    if RAND_TYPE == FLOAT_TYPE:
        return numpy.random.normal(0, 5, test_size)
    elif RAND_TYPE == INT64_TYPE:
        return numpy.random.randint(-2 ** 10, 2 ** 10, test_size)
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
        try:
            assert_diff(res[i], ref[i])
        except AssertionError:
            print("Assertion Error at location", i, ", FPGA result:",
                  res[i], ", reference result:", ref[i])


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        return res, end_time - start_time

    return wrapper


def compare_time(fpga_time, cpu_time, num_instances=TEST_SIZE):
    print("FPGA time:", fpga_time, "second(s)")
    print("CPU time:", cpu_time, "second(s)")
    print("FPGA throughput:", num_instances / fpga_time, "instance(s) per second")
    print("CPU throughput:", num_instances / cpu_time, "instance(s) per second")
    print("Speedup:", cpu_time / fpga_time)


def cpu_pi_gen_obf_seed(res_store, public_key, count, elem_size, rand_seed, stream):
    random.seed(rand_seed)
    rand_vals = [random.randrange(1, 8 ** elem_size) for _ in range(count)]
    return [gmpy_math.powmod(v, public_key.n, public_key.nsquare) for v in rand_vals]


def cpu_pi_obfuscate(public_key, encrypted_numbers, obf_seeds, exponents, res_store, stream):
    return [PaillierEncryptedNumber(public_key, (encrypted_numbers[i] * obf_seeds[i]) % public_key.nsquare,
                                    exponents[i]) for i in range(len(encrypted_numbers))]


def cpu_fp_mul(left, right):
    return [FixedPointNumber((left[i].encoding * right[i].encoding) % left[i].n,
                             left[i].exponent + right[i].exponent, left[i].n, left[i].max_int) for i in
            range(len(left))]


class TestOperators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        initialize_device()
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        cls.n, cls.max_int = cls._pub_key.n, cls._pub_key.max_int
        cls._cpu_pub_key = pi_p2c_pub_key(None, cls._pub_key)
        cls._cpu_priv_key = pi_p2c_priv_key(None, cls._priv_key)
        cls._fpga_pub_key = pi_h2d_pub_key(None, cls._cpu_pub_key)
        cls._fpga_priv_key = pi_h2d_priv_key(None, cls._cpu_priv_key)
        print("\n\n", "*" * 100, "\n\nInitialization complete\nTest Size:", TEST_SIZE)

    # test performance
    def test_performance(self):
        print("\n\n", "*" * 100, "\n\nTest performance begins")

        print("\n>>>>> generate data and allocate memory spaces")
        raw, raw2 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        shape_tuple, shape_tuple_T = (NUM_ROWS, NUM_COLS), (NUM_COLS, NUM_ROWS)
        shape_store, shape_store_T = TensorShapeStorage(*shape_tuple), TensorShapeStorage(*shape_tuple_T)
        fpga_bi_store, fpga_bi_store2 = bi_alloc(
            None, TEST_SIZE, PLAIN_BYTE, MEM_HOST), bi_alloc(
            None, TEST_SIZE, PLAIN_BYTE, MEM_HOST)
        fpga_te_store, fpga_te_store2 = te_alloc(None, TEST_SIZE, MEM_HOST), te_alloc(None, TEST_SIZE, MEM_HOST)
        fpga_fp_store, fpga_fp_store2 = fp_alloc(None, TEST_SIZE, MEM_HOST), fp_alloc(None, TEST_SIZE, MEM_HOST)
        fpga_pi_store, fpga_pi_store2 = pi_alloc(None, TEST_SIZE, MEM_HOST), pi_alloc(None, TEST_SIZE, MEM_HOST)
        fpga_te_store, fpga_te_store2 = te_p2c(raw, fpga_te_store), te_p2c(raw2, fpga_te_store2)

        print("\n>>>>> fp_encode profiling begins")
        fpga_encoded, fpga_encode_time = profile(fp_encode)(fpga_te_store, self.n, self.max_int, res=fpga_fp_store)
        cpu_encoded, cpu_encode_time = profile(
            lambda l: [
                FixedPointNumber.encode(
                    v, self.n, self.max_int) for v in l])(raw)
        compare_time(fpga_encode_time, cpu_encode_time)

        print("\n>>>>> fp_decode profiling begins")
        fpga_decoded, fpga_decode_time = profile(fp_decode)(fpga_encoded, fpga_te_store, None)
        cpu_decoded, cpu_decode_time = profile(lambda l: [v.decode() for v in l])(cpu_encoded)
        compare_time(fpga_decode_time, cpu_decode_time)

        # check decoded results
        assert_ndarray_diff(te_c2p(fpga_decoded), numpy.asarray(cpu_decoded))

        print("\n>>>>> pi_encrypt profiling begins")
        print("This function calculates (encoding * n + 1) % nsquare")
        fpga_encrypted, fpga_encrypt_time = profile(pi_encrypt)(self._fpga_pub_key, fpga_encoded, fpga_pi_store, None)
        cpu_encrypted, cpu_encrypt_time = profile(
            lambda l: [
                self._pub_key.raw_encrypt(
                    v.encoding, 1) for v in l])(cpu_encoded)
        compare_time(fpga_encrypt_time, cpu_encrypt_time)

        print("\n>>>>> pi_gen_obf_seed profiling begins")
        print("This function calculates (rand() ^ n) % nsquare")
        fpga_obf_seeds, fpga_gen_obf_seeds_time = profile(pi_gen_obf_seed)(
            fpga_bi_store, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, 0, None)
        cpu_obf_seeds, cpu_gen_obf_seefs_time = profile(cpu_pi_gen_obf_seed)(
            None, self._pub_key, TEST_SIZE, CIPHER_BITS // 6, 0, None)
        compare_time(fpga_gen_obf_seeds_time, cpu_gen_obf_seefs_time)

        print("\n>>>>> pi_obfuscate profiling begins")
        print("This function calculates (raw_cipher * obf_seed) % nsquare,")
        print("\twhere raw_cipher and obf_seed are calculated in pi_encrypt and pi_gen_obf_seeds, respectively")
        fpga_obfuscated, fpga_obfuscate_time = profile(pi_obfuscate)(
            self._fpga_pub_key, fpga_encrypted, fpga_obf_seeds, fpga_pi_store, None)
        cpu_obfuscated, cpu_obfuscate_time = profile(cpu_pi_obfuscate)(
            self._pub_key, cpu_encrypted, cpu_obf_seeds, [
                v.exponent for v in cpu_encoded], None, None)
        compare_time(fpga_obfuscate_time, cpu_obfuscate_time)

        # check intermediate result
        assert_ndarray_diff(numpy.asarray(pi_c2p(fpga_obfuscated)[0]), numpy.asarray(
            [v.ciphertext(False) for v in cpu_obfuscated]))

        print("\n>>>>> pi_decrypt profiling begins")
        print("This function calculates L(cipher ^ lambda % nsquare) * L(g ^ lambda % nsquare) ^ -1 % n")
        print("fp_decode is by default included in pi_decrypt")
        fps_buffer = fp_alloc(None, TEST_SIZE, MEM_HOST)
        fpga_decrypted, fpga_decrypt_time = profile(pi_decrypt)(
            self._fpga_pub_key, self._fpga_priv_key, fpga_obfuscated, fpga_te_store, fps_buffer)
        cpu_decrypted, cpu_decrypt_time = profile(lambda l: [self._priv_key.decrypt(v) for v in l])(cpu_obfuscated)
        compare_time(fpga_decrypt_time, cpu_decrypt_time)

        # check decrypted results
        assert_ndarray_diff(te_c2p(fpga_decrypted), numpy.asarray(cpu_decrypted))

        print("\n>>>>> generating the other array")
        # encode the other array
        fpga_encoded2 = fp_encode(fpga_te_store2, self.n, self.max_int, res=fpga_fp_store2)
        cpu_encoded2 = [FixedPointNumber.encode(v, self.n, self.max_int) for v in raw2]
        # encrypt the other array
        fpga_encrypted2 = pi_encrypt(self._fpga_pub_key, fpga_encoded2, fpga_pi_store2, None)
        cpu_encrypted2 = [self._pub_key.raw_encrypt(v.encoding, 1) for v in cpu_encoded2]
        # generate obfuscation seeds (obfuscators) for the other array using a different random seed
        fpga_obf_seeds2 = pi_gen_obf_seed(fpga_bi_store2, self._fpga_pub_key, TEST_SIZE, CIPHER_BITS // 6, 1, None)
        cpu_obf_seeds2 = cpu_pi_gen_obf_seed(None, self._pub_key, TEST_SIZE, CIPHER_BITS // 6, 1, None)
        # obfuscate the other array
        fpga_obfuscated2 = pi_obfuscate(self._fpga_pub_key, fpga_encrypted2, fpga_obf_seeds2, fpga_pi_store2, None)
        cpu_obfuscated2 = cpu_pi_obfuscate(
            self._pub_key, cpu_encrypted2, cpu_obf_seeds2, [
                v.exponent for v in cpu_encoded2], None, None)
        # check intermediate result
        assert_ndarray_diff(numpy.asarray(pi_c2p(fpga_obfuscated2)[0]), numpy.asarray(
            [v.ciphertext(False) for v in cpu_obfuscated2]))

        print("\n>>>>> fp_mul profiling begins")
        fpga_fp_mul_store = fp_alloc(None, TEST_SIZE, MEM_HOST)
        (fpga_fp_mul_res, _), fpga_fp_mul_time = profile(fp_mul)(fpga_encoded,
                                                                 fpga_encoded2, shape_store, shape_store,
                                                                 fpga_fp_mul_store, shape_store, None)
        cpu_fp_mul_res, cpu_fp_mul_time = profile(cpu_fp_mul)(cpu_encoded, cpu_encoded2)
        compare_time(fpga_fp_mul_time, cpu_fp_mul_time)

        # Compare results
        received_fp_mul_res = fp_c2p(fpga_fp_mul_res)
        for i in range(TEST_SIZE):
            assert_diff(received_fp_mul_res[i].encoding, cpu_fp_mul_res[i].encoding)
            assert received_fp_mul_res[i].BASE == cpu_fp_mul_res[i].BASE
            assert received_fp_mul_res[i].exponent == cpu_fp_mul_res[i].exponent

        print("\n>>>>> pi_add profiling begins")
        (fpga_add_res, _), fpga_add_time = profile(pi_add)(self._fpga_pub_key, fpga_obfuscated,
                                                           fpga_obfuscated2, shape_store, shape_store, fpga_pi_store,
                                                           shape_store, None)
        cpu_add_res, cpu_add_time = profile(lambda a, b: [a[i] + b[i]
                                                          for i in range(TEST_SIZE)])(cpu_obfuscated, cpu_obfuscated2)
        compare_time(fpga_add_time, cpu_add_time)

        print("\n>>>>> pi_mul profiling begins")
        (fpga_mul_res, _), fpga_mul_time = profile(pi_mul)(self._fpga_pub_key, fpga_add_res,
                                                           fpga_encoded2, shape_store, shape_store, fpga_pi_store,
                                                           shape_store, None)
        cpu_mul_res, cpu_mul_time = profile(lambda a, b: [a[i] * b[i]
                                                          for i in range(TEST_SIZE)])(cpu_add_res, cpu_encoded2)
        compare_time(fpga_mul_time, cpu_mul_time)

        print("\n>>>>> pi_matmul profiling begins")
        print("sizes are", shape_tuple, "and", shape_tuple_T)
        fpga_pi_matmul_store = pi_alloc(None, NUM_ROWS * NUM_ROWS, MEM_HOST)
        (fpga_matmul_res, fpga_matmul_shape), fpga_matmul_time = profile(pi_matmul)(self._fpga_pub_key,
                                                                                    fpga_mul_res, fpga_encoded2,
                                                                                    shape_store, shape_store_T,
                                                                                    fpga_pi_matmul_store, None, None)
        cpu_matmul_res, cpu_matmul_time = profile(
            lambda a, b: a @ b)(numpy.asarray(cpu_mul_res).reshape(shape_tuple),
                                numpy.asarray(cpu_encoded2).reshape(shape_tuple_T))
        compare_time(fpga_matmul_time, cpu_matmul_time, NUM_ROWS * TEST_SIZE)

        print("\n>>>>> pi_sum profiling begins")
        print("shape is", fpga_matmul_shape.to_tuple())
        fpga_pi_sum_store = pi_alloc(None, max(NUM_ROWS, NUM_COLS), MEM_HOST)
        for axis in [0, 1, None]:
            print(">>> axis:", axis)
            (fpga_sum_res, _), fpga_sum_time = profile(pi_sum)(self._fpga_pub_key,
                                                               fpga_matmul_res, fpga_matmul_shape, axis,
                                                               fpga_pi_sum_store, None, None)
            cpu_sum_res, cpu_sum_time = profile(lambda a: numpy.sum(a, axis))(cpu_matmul_res)
            compare_time(fpga_sum_time, cpu_sum_time)

            # check result
            fpga_decrypted = te_c2p(pi_decrypt(self._fpga_pub_key, self._fpga_priv_key, fpga_sum_res, None, None))
            cpu_decrypted = numpy.asarray([self._priv_key.decrypt(v) for v in cpu_sum_res.flat]
                                          if axis is not None else [self._priv_key.decrypt(cpu_sum_res)])
            assert_ndarray_diff(fpga_decrypted, cpu_decrypted)

        print("\n>>>>> free all allocated spaces")
        bi_free(fpga_bi_store)
        bi_free(fpga_bi_store2)
        te_free(fpga_te_store)
        te_free(fpga_te_store2)
        fp_free(fpga_fp_store)
        fp_free(fpga_fp_store2)
        fp_free(fps_buffer)
        fp_free(fpga_fp_mul_store)
        pi_free(fpga_pi_store)
        pi_free(fpga_pi_store2)
        pi_free(fpga_pi_matmul_store)
        pi_free(fpga_pi_sum_store)

        print("test passed")


if __name__ == "__main__":
    unittest.main()
