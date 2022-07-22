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

import functools
import time
import unittest
import numpy
from fate_arch.tensor.impl.blocks.python_paillier_block import FixedPointNumber, PaillierKeypair

from ..gpu_engine import (
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
    pi_decrypt,
    fp_mul,
    fp_c2p,
    pi_add,
    pi_mul,
    pi_sum,
    bi_free,
    te_free,
    fp_free,
    pi_free,
    initialize_device,
    pi_matmul,
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
            print("Assertion Error at location", i, ", GPU result:",
                  res[i], ", reference result:", ref[i])


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        return res, end_time - start_time

    return wrapper


def compare_time(gpu_time, cpu_time, num_instances=TEST_SIZE):
    print("GPU time:", gpu_time, "second(s)")
    print("CPU time:", cpu_time, "second(s)")
    print("GPU throughput:", num_instances / gpu_time, "instance(s) per second")
    print("CPU throughput:", num_instances / cpu_time, "instance(s) per second")
    print("Speedup:", cpu_time / gpu_time)


class TestGPUPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        initialize_device()
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        cls.n, cls.max_int = cls._pub_key.n, cls._pub_key.max_int
        cls._cpu_pub_key = pi_p2c_pub_key(None, cls._pub_key)
        cls._cpu_priv_key = pi_p2c_priv_key(None, cls._priv_key)
        cls._gpu_pub_key = pi_h2d_pub_key(None, cls._cpu_pub_key)
        cls._gpu_priv_key = pi_h2d_priv_key(None, cls._cpu_priv_key)
        print("\n\n", "*" * 100, "\n\nInitialization complete\nTest Size:", TEST_SIZE)

    # test performance
    def test_performance(self):
        print("\n\n", "*" * 100, "\n\nTest performance begins")

        print("\n>>>>> generate data and allocate memory spaces")
        raw, raw2 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        shape_tuple, shape_tuple_T = (NUM_ROWS, NUM_COLS), (NUM_COLS, NUM_ROWS)
        shape_store, shape_store_T = TensorShapeStorage(*shape_tuple), TensorShapeStorage(*shape_tuple_T)
        gpu_bi_store, gpu_bi_store2 = bi_alloc(None, TEST_SIZE, PLAIN_BYTE, MEM_HOST), bi_alloc(None, TEST_SIZE,
                                                                                                PLAIN_BYTE, MEM_HOST)
        gpu_te_store, gpu_te_store2 = te_alloc(None, TEST_SIZE, MEM_HOST), te_alloc(None, TEST_SIZE, MEM_HOST)
        gpu_fp_store, gpu_fp_store2 = fp_alloc(None, TEST_SIZE, MEM_HOST), fp_alloc(None, TEST_SIZE, MEM_HOST)
        gpu_pi_store, gpu_pi_store2 = pi_alloc(None, TEST_SIZE, MEM_HOST), pi_alloc(None, TEST_SIZE, MEM_HOST)
        gpu_te_store, gpu_te_store2 = te_p2c(raw, gpu_te_store), te_p2c(raw2, gpu_te_store2)

        print("\n>>>>> fp_encode profiling begins")
        gpu_encoded, gpu_encode_time = profile(fp_encode)(gpu_te_store, self.n, self.max_int, res=gpu_fp_store)
        cpu_encode_time = TEST_SIZE / 62303.97
        compare_time(gpu_encode_time, cpu_encode_time)

        print("\n>>>>> fp_decode profiling begins")
        gpu_decoded, gpu_decode_time = profile(fp_decode)(gpu_encoded, gpu_te_store, None)
        cpu_decode_time = TEST_SIZE / 567913.21
        compare_time(gpu_decode_time, cpu_decode_time)

        # check decoded results
        assert_ndarray_diff(te_c2p(gpu_decoded), numpy.asarray(raw))

        print("\n>>>>> pi_encrypt profiling begins")
        print("This function calculates (encoding * n + 1) % nsquare")
        gpu_encrypted, gpu_encrypt_time = profile(pi_encrypt)(self._gpu_pub_key, gpu_encoded, gpu_pi_store, None)
        cpu_encrypt_time = TEST_SIZE / 205864.74
        compare_time(gpu_encrypt_time, cpu_encrypt_time)

        print("\n>>>>> pi_gen_obf_seed profiling begins")
        print("This function calculates (rand() ^ n) % nsquare")
        gpu_obf_seeds, gpu_gen_obf_seeds_time = profile(pi_gen_obf_seed)(gpu_bi_store, self._gpu_pub_key, TEST_SIZE,
                                                                         CIPHER_BITS // 6, 0, None)
        cpu_gen_obf_seefs_time = TEST_SIZE / 444.05
        compare_time(gpu_gen_obf_seeds_time, cpu_gen_obf_seefs_time)

        print("\n>>>>> pi_obfuscate profiling begins")
        print("This function calculates (raw_cipher * obf_seed) % nsquare,")
        print("\twhere raw_cipher and obf_seed are calculated in pi_encrypt and pi_gen_obf_seeds, respectively")
        gpu_obfuscated, gpu_obfuscate_time = profile(pi_obfuscate)(self._gpu_pub_key, gpu_encrypted, gpu_obf_seeds,
                                                                   gpu_pi_store, None)
        cpu_obfuscate_time = TEST_SIZE / 60236.27
        compare_time(gpu_obfuscate_time, cpu_obfuscate_time)

        print("\n>>>>> pi_decrypt profiling begins")
        print("This function calculates L(cipher ^ lambda % nsquare) * L(g ^ lambda % nsquare) ^ -1 % n")
        print("fp_decode is by default included in pi_decrypt")
        fps_buffer = fp_alloc(None, TEST_SIZE, MEM_HOST)
        gpu_decrypted, gpu_decrypt_time = profile(pi_decrypt)(self._gpu_pub_key, self._gpu_priv_key, gpu_obfuscated,
                                                              gpu_te_store, fps_buffer)
        cpu_decrypt_time = TEST_SIZE / 1590.48
        compare_time(gpu_decrypt_time, cpu_decrypt_time)

        # check decrypted results
        assert_ndarray_diff(te_c2p(gpu_decrypted), numpy.asarray(raw))

        print("\n>>>>> generating the other array")
        gpu_encoded2 = fp_encode(gpu_te_store2, self.n, self.max_int, res=gpu_fp_store2)
        gpu_encrypted2 = pi_encrypt(self._gpu_pub_key, gpu_encoded2, gpu_pi_store2, None)
        gpu_obf_seeds2 = pi_gen_obf_seed(gpu_bi_store2, self._gpu_pub_key, TEST_SIZE, CIPHER_BITS // 6, 1, None)
        gpu_obfuscated2 = pi_obfuscate(self._gpu_pub_key, gpu_encrypted2, gpu_obf_seeds2, gpu_pi_store2, None)

        print("\n>>>>> fp_mul profiling begins")
        gpu_fp_mul_store = fp_alloc(None, TEST_SIZE, MEM_HOST)
        (gpu_fp_mul_res, _), gpu_fp_mul_time = profile(fp_mul)(gpu_encoded, gpu_encoded2, shape_store, shape_store,
                                                               gpu_fp_mul_store, shape_store, None)
        cpu_fp_mul_time = TEST_SIZE / 228424.79
        compare_time(gpu_fp_mul_time, cpu_fp_mul_time)

        # Compare results
        cpu_encoded = [FixedPointNumber.encode(v, self.n, self.max_int) for v in raw]
        cpu_encoded2 = [FixedPointNumber.encode(v, self.n, self.max_int) for v in raw2]
        cpu_fp_mul_res = [FixedPointNumber((cpu_encoded[i].encoding * cpu_encoded2[i].encoding) % cpu_encoded[i].n,
                                           cpu_encoded[i].exponent + cpu_encoded2[i].exponent, cpu_encoded[i].n,
                                           cpu_encoded[i].max_int)
                          for i in range(TEST_SIZE)]
        received_fp_mul_res = fp_c2p(gpu_fp_mul_res)
        for i in range(TEST_SIZE):
            assert_diff(received_fp_mul_res[i].encoding, cpu_fp_mul_res[i].encoding)
            assert received_fp_mul_res[i].BASE == cpu_fp_mul_res[i].BASE
            assert received_fp_mul_res[i].exponent == cpu_fp_mul_res[i].exponent

        print("\n>>>>> pi_add profiling begins")
        (gpu_add_res, _), gpu_add_time = profile(pi_add)(self._gpu_pub_key, gpu_obfuscated, gpu_obfuscated2,
                                                         shape_store, shape_store, gpu_pi_store, shape_store, None)
        cpu_add_time = TEST_SIZE / 29759.90
        compare_time(gpu_add_time, cpu_add_time)

        print("\n>>>>> pi_mul profiling begins")
        (gpu_mul_res, _), gpu_mul_time = profile(pi_mul)(self._gpu_pub_key, gpu_add_res, gpu_encoded2, shape_store,
                                                         shape_store, gpu_pi_store, shape_store, None)
        cpu_mul_time = TEST_SIZE / 6175.70
        compare_time(gpu_mul_time, cpu_mul_time)

        print("\n>>>>> pi_matmul profiling begins")
        print("sizes are", shape_tuple, "and", shape_tuple_T)
        gpu_pi_matmul_store = pi_alloc(None, NUM_ROWS * NUM_ROWS, MEM_HOST)
        (gpu_matmul_res, gpu_matmul_shape), gpu_matmul_time = profile(pi_matmul)(self._gpu_pub_key, gpu_mul_res,
                                                                                 gpu_encoded2, shape_store,
                                                                                 shape_store_T, gpu_pi_matmul_store,
                                                                                 None, None)
        cpu_matmul_time = NUM_ROWS * TEST_SIZE / 4178.43
        compare_time(gpu_matmul_time, cpu_matmul_time, NUM_ROWS * TEST_SIZE)

        print("\n>>>>> pi_sum profiling begins")
        print("shape is", gpu_matmul_shape.to_tuple())
        gpu_pi_sum_store = pi_alloc(None, max(NUM_ROWS, NUM_COLS), MEM_HOST)
        decrypted_matmul_res = numpy.asarray(
            te_c2p(pi_decrypt(self._gpu_pub_key, self._gpu_priv_key, gpu_matmul_res, None, None))).reshape(
            gpu_matmul_shape.to_tuple())
        for axis in [0, 1, None]:
            print(">>> axis:", axis)
            (gpu_sum_res, _), gpu_sum_time = profile(pi_sum)(self._gpu_pub_key, gpu_matmul_res, gpu_matmul_shape, axis,
                                                             gpu_pi_sum_store, None, None)
            cpu_sum_time = TEST_SIZE / (12865.10 if axis == 0 else (15919.62 if axis == 1 else 10277.66))
            compare_time(gpu_sum_time, cpu_sum_time)

            # check result
            gpu_decrypted = te_c2p(pi_decrypt(self._gpu_pub_key, self._gpu_priv_key, gpu_sum_res, None, None))
            cpu_sum = decrypted_matmul_res.sum(axis)
            if axis is None:
                cpu_sum = numpy.asarray([cpu_sum])
            assert_ndarray_diff(gpu_decrypted, cpu_sum)

        print("\n>>>>> free all allocated spaces")
        bi_free(gpu_bi_store)
        bi_free(gpu_bi_store2)
        te_free(gpu_te_store)
        te_free(gpu_te_store2)
        fp_free(gpu_fp_store)
        fp_free(gpu_fp_store2)
        fp_free(fps_buffer)
        fp_free(gpu_fp_mul_store)
        pi_free(gpu_pi_store)
        pi_free(gpu_pi_store2)
        pi_free(gpu_pi_matmul_store)
        pi_free(gpu_pi_sum_store)

        print("test passed")


if __name__ == "__main__":
    unittest.main()
