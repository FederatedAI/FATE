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
import random

import numpy as np
import unittest
import functools
import time

from fate_tensor_gpu.secureprotol.fixedpoint import FixedPointNumber
from fate_tensor_gpu.secureprotol import gmpy_math
from fate_tensor_gpu.secureprotol.fate_paillier import (
    PaillierKeypair,
    PaillierEncryptedNumber,
)

from fate_tensor_gpu.gpu_engine import (
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
    pi_free,
)

RAND_TYPE = FLOAT_TYPE  # SWITCH DATA TYPE HERE: EITHER INT64_TYPE OR FLOAT_TYPE
NUM_ROWS = 200
NUM_COLS = 200
TEST_SIZE = NUM_ROWS * NUM_COLS
KEY_LEN = 1024
DATA_SIZE = TEST_SIZE * KEY_LEN * 2 // 8
ERROR_TOLERANCE = 1e-10


class TestCaseReport:
    def __init__(self, name, batch_size, bit_len, data_size):
        self.name = name
        self.batch_size = batch_size
        self.bit_len = bit_len
        self.data_size = int(data_size)
        self.content = {}
        self.width = 100
        self.column = [30, 20, 25, 24]
        self.cpu_throughput = 0.0
        self.gpu_throughput = 0.0

    def add_perf_report(self, name):
        self.content[name] = {}

    def add_item(self, report_name, item_name, time, ops, bw):
        self.content[report_name][item_name] = {}
        self.content[report_name][item_name]['time'] = time
        self.content[report_name][item_name]['ops'] = ops
        self.content[report_name][item_name]['bw'] = bw

    def gen_line(self, *args):
        i = 0
        size = 0
        res = ''
        for v in args:
            res += '|' + str(v) + ' ' * (self.column[i] - len(str(v)) - 1)
            size += self.column[i]
            i += 1
        if i < 3:
            res += " " * (self.width - size - 1)
        res += '|'
        return res

    def dump_header(self):
        res = []
        res.append('=' * self.width)
        res.append(
            '|' + ' ' * (int(self.width - len(self.name) - 2) // 2) + self.name + ' ' * (
                int(self.width - len(self.name) - 1) // 2) + '|'
        )
        res.append('=' * self.width)
        res.append(self.gen_line("Data Information"))
        res.append('-' * self.width)
        res.append(self.gen_line("Batch Size", self.batch_size))
        res.append(self.gen_line("Bit Length", self.bit_len))
        res.append(self.gen_line("Data Size (Bytes)", self.data_size))
        return "\n".join(res)

    def dump_item(self, report_name, item_name):
        time = self.content[report_name][item_name]['time']
        time = "{0:.4f}".format(time)
        ops = self.content[report_name][item_name]['ops']
        ops = "{0:.4f}".format(ops)
        bw = self.content[report_name][item_name]['bw'] / (2 ** 20)
        bw = "{0:.4f}".format(bw)
        line = self.gen_line(item_name, time, ops, bw)
        return line

    def dump_perf_report(self, report_name):
        res = []
        res.append("=" * self.width)
        res.append(self.gen_line(report_name))
        res.append("-" * self.width)
        res.append(
            self.gen_line(
                "Item",
                "Time Elapsed(s)",
                "Operations Per Second",
                "Bandwidth (MB/s)"))
        res.append("-" * self.width)
        for v in self.content[report_name]:
            res.append(self.dump_item(report_name, v))
        return "\n".join(res)

    def dump_summary(self):
        self.ratio = self.gpu_throughput / self.cpu_throughput
        res = []
        res.append("=" * self.width)
        res.append(self.gen_line("Performance of GPU/CPU"))
        res.append('-' * self.width)
        res.append(
            self.gen_line(
                "GPU/CPU Ratio (Speedup)",
                "{0:.4f}".format(
                    self.ratio)))
        res.append("=" * self.width)
        res.append('\n')

        return "\n".join(res)

    def dump_result(self):
        res = []
        res.append(self.dump_header())
        for v in self.content:
            res.append(self.dump_perf_report(v))
        res.append(self.dump_summary())
        report = "\n".join(res)
        print(report)


def generate_rand(test_size):
    if RAND_TYPE == FLOAT_TYPE:
        return np.random.normal(0, 5, test_size)
    elif RAND_TYPE == INT64_TYPE:
        return np.random.randint(-(2 ** 10), 2 ** 10, test_size)
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
            print(
                "Assertion Error at location",
                i,
                ", GPU result:",
                res[i],
                ", reference result:",
                ref[i],
            )


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
    print(
        "GPU throughput:",
        num_instances / gpu_time,
        "instance(s) per second")
    print(
        "CPU throughput:",
        num_instances / cpu_time,
        "instance(s) per second")
    print("Speedup:", cpu_time / gpu_time)


def cpu_pi_gen_obf_seed(
        res_store,
        public_key,
        count,
        elem_size,
        rand_seed,
        stream):
    random.seed(rand_seed)
    rand_vals = [random.randrange(1, 8 ** elem_size) for _ in range(count)]
    return [
        gmpy_math.powmod(
            v,
            public_key.n,
            public_key.nsquare) for v in rand_vals]


def cpu_pi_obfuscate(
        public_key, encrypted_numbers, obf_seeds, exponents, res_store, stream
):
    return [
        PaillierEncryptedNumber(
            public_key,
            (encrypted_numbers[i] * obf_seeds[i]) % public_key.nsquare,
            exponents[i],
        )
        for i in range(len(encrypted_numbers))
    ]


def cpu_fp_mul(left, right):
    return [
        FixedPointNumber(
            (left[i].encoding * right[i].encoding) % left[i].n,
            left[i].exponent + right[i].exponent,
            left[i].n,
            left[i].max_int,
        )
        for i in range(len(left))
    ]


def add_to_perf_reports(_perf_reports, name, gpu_time, cpu_time, data_size):
    perf_report = TestCaseReport(name, TEST_SIZE, KEY_LEN, data_size)
    perf_report.gpu_throughput = TEST_SIZE / gpu_time
    perf_report.add_perf_report("GPU Performance")
    perf_report.add_item(
        "GPU Performance",
        "Computation on GPU",
        gpu_time,
        TEST_SIZE / gpu_time,
        data_size / gpu_time,
    )
    perf_report.cpu_throughput = TEST_SIZE / cpu_time
    perf_report.add_perf_report("CPU Performance")
    perf_report.add_item(
        "CPU Performance",
        "Computation on CPU",
        cpu_time,
        TEST_SIZE / cpu_time,
        data_size / cpu_time,
    )
    _perf_reports.append(perf_report)


class TestOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        cls.n, cls.max_int = cls._pub_key.n, cls._pub_key.max_int
        cls._cpu_pub_key = pi_p2c_pub_key(cls._pub_key)
        cls._cpu_priv_key = pi_p2c_priv_key(cls._priv_key)
        cls._gpu_pub_key = pi_h2d_pub_key(cls._cpu_pub_key)
        cls._gpu_priv_key = pi_h2d_priv_key(cls._cpu_priv_key)
        cls._perf_reports = []
        print(
            "\n\n",
            "*" * 100,
            "\n\nInitialization complete\nTest Size:",
            TEST_SIZE)

    def test_performance(self):
        print("\n\n", "*" * 100, "\n\nTest performance begins")

        print("\n>>>>> generate data and allocate memory spaces")
        raw, raw2 = generate_rand(TEST_SIZE), generate_rand(TEST_SIZE)
        shape_tuple, shape_tuple_T = (NUM_ROWS, NUM_COLS), (NUM_COLS, NUM_ROWS)
        shape_store, _ = TensorShapeStorage(*shape_tuple), TensorShapeStorage(
            *shape_tuple_T
        )
        gpu_bi_store, gpu_bi_store2 = bi_alloc(
            None, TEST_SIZE, PLAIN_BYTE, MEM_HOST
        ), bi_alloc(None, TEST_SIZE, PLAIN_BYTE, MEM_HOST)
        gpu_te_store, gpu_te_store2 = te_alloc(
            None, TEST_SIZE, MEM_HOST), te_alloc(
            None, TEST_SIZE, MEM_HOST)
        gpu_fp_store, gpu_fp_store2 = fp_alloc(
            None, TEST_SIZE, MEM_HOST), fp_alloc(
            None, TEST_SIZE, MEM_HOST)
        gpu_pi_store, gpu_pi_store2 = pi_alloc(
            None, TEST_SIZE, MEM_HOST), pi_alloc(
            None, TEST_SIZE, MEM_HOST)
        gpu_te_store, gpu_te_store2 = te_p2c(raw, gpu_te_store), te_p2c(
            raw2, gpu_te_store2
        )

        print("\n>>>>> fp_encode profiling begins")
        gpu_encoded, gpu_encode_time = profile(fp_encode)(
            gpu_te_store, self.n, self.max_int, res=gpu_fp_store
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_encode_time
            )
        )
        cpu_encoded, cpu_encode_time = profile(
            lambda l: [
                FixedPointNumber.encode(
                    v, self.n, self.max_int) for v in l])(raw)
        compare_time(gpu_encode_time, cpu_encode_time)
        add_to_perf_reports(
            self._perf_reports,
            "Encode",
            gpu_encode_time,
            cpu_encode_time,
            DATA_SIZE)

        print("\n>>>>> fp_decode profiling begins")
        gpu_decoded, gpu_decode_time = profile(fp_decode)(
            gpu_encoded, gpu_te_store, None
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_decode_time
            )
        )
        cpu_decoded, cpu_decode_time = profile(
            lambda l: [v.decode() for v in l])(cpu_encoded)
        compare_time(gpu_decode_time, cpu_decode_time)
        add_to_perf_reports(
            self._perf_reports,
            "Decode",
            gpu_decode_time,
            cpu_decode_time,
            DATA_SIZE)

        # check decoded results
        assert_ndarray_diff(te_c2p(gpu_decoded), np.asarray(cpu_decoded))

        print("\n>>>>> pi_encrypt profiling begins")
        print("This function calculates (encoding * n + 1) % nsquare")
        gpu_encrypted, gpu_encrypt_time = profile(pi_encrypt)(
            self._gpu_pub_key, gpu_encoded, gpu_pi_store, None
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_encrypt_time
            )
        )
        cpu_encrypted, cpu_encrypt_time = profile(
            lambda l: [self._pub_key.raw_encrypt(v.encoding, 1) for v in l]
        )(cpu_encoded)
        compare_time(gpu_encrypt_time, cpu_encrypt_time)
        add_to_perf_reports(
            self._perf_reports,
            "Encrypt",
            gpu_encrypt_time,
            cpu_encrypt_time,
            DATA_SIZE)

        print("\n>>>>> pi_gen_obf_seed profiling begins")
        print("This function calculates (rand() ^ n) % nsquare")
        gpu_obf_seeds, gpu_gen_obf_seeds_time = profile(pi_gen_obf_seed)(
            gpu_bi_store, self._gpu_pub_key, TEST_SIZE, CIPHER_BITS // 6, 0, None)
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_gen_obf_seeds_time
            )
        )
        cpu_obf_seeds, cpu_gen_obf_seefs_time = profile(cpu_pi_gen_obf_seed)(
            None, self._pub_key, TEST_SIZE, CIPHER_BITS // 6, 0, None
        )
        compare_time(gpu_gen_obf_seeds_time, cpu_gen_obf_seefs_time)
        add_to_perf_reports(
            self._perf_reports,
            "Generate Obfuscators",
            gpu_gen_obf_seeds_time,
            cpu_gen_obf_seefs_time,
            DATA_SIZE,
        )

        print("\n>>>>> pi_obfuscate profiling begins")
        print("This function calculates (raw_cipher * obf_seed) % nsquare,")
        print(
            "\twhere raw_cipher and obf_seed are calculated in pi_encrypt and pi_gen_obf_seeds, respectively"
        )
        gpu_obfuscated, gpu_obfuscate_time = profile(pi_obfuscate)(
            self._gpu_pub_key, gpu_encrypted, gpu_obf_seeds, gpu_pi_store, None
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_obfuscate_time
            )
        )
        cpu_obfuscated, cpu_obfuscate_time = profile(cpu_pi_obfuscate)(
            self._pub_key,
            cpu_encrypted,
            cpu_obf_seeds,
            [v.exponent for v in cpu_encoded],
            None,
            None,
        )
        compare_time(gpu_obfuscate_time, cpu_obfuscate_time)
        add_to_perf_reports(
            self._perf_reports,
            "Obfuscate",
            gpu_obfuscate_time,
            cpu_obfuscate_time,
            DATA_SIZE,
        )

        # check intermediate result
        assert_ndarray_diff(
            np.asarray(pi_c2p(gpu_obfuscated)[0]),
            np.asarray([v.ciphertext(False) for v in cpu_obfuscated]),
        )

        print("\n>>>>> pi_decrypt profiling begins")
        print(
            "This function calculates L(cipher ^ lambda % nsquare) * L(g ^ lambda % nsquare) ^ -1 % n"
        )
        print("fp_decode is by default included in pi_decrypt")
        fps_buffer = fp_alloc(None, TEST_SIZE, MEM_HOST)
        gpu_decrypted, gpu_decrypt_time = profile(pi_decrypt)(
            self._gpu_pub_key,
            self._gpu_priv_key,
            gpu_obfuscated,
            gpu_te_store,
            fps_buffer,
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_decrypt_time
            )
        )
        cpu_decrypted, cpu_decrypt_time = profile(
            lambda l: [self._priv_key.decrypt(v) for v in l]
        )(cpu_obfuscated)
        compare_time(gpu_decrypt_time, cpu_decrypt_time)
        add_to_perf_reports(
            self._perf_reports,
            "Decrypt",
            gpu_decrypt_time,
            cpu_decrypt_time,
            DATA_SIZE)

        # check decrypted results
        assert_ndarray_diff(te_c2p(gpu_decrypted), np.asarray(cpu_decrypted))

        print("\n>>>>> generating the other array")
        # encode the other array
        gpu_encoded2 = fp_encode(
            gpu_te_store2,
            self.n,
            self.max_int,
            res=gpu_fp_store2)
        cpu_encoded2 = [
            FixedPointNumber.encode(
                v, self.n, self.max_int) for v in raw2]
        # encrypt the other array
        gpu_encrypted2 = pi_encrypt(
            self._gpu_pub_key, gpu_encoded2, gpu_pi_store2, None
        )
        cpu_encrypted2 = [
            self._pub_key.raw_encrypt(v.encoding, 1) for v in cpu_encoded2
        ]
        # generate obfuscation seeds (obfuscators) for the other array using a
        # different random seed
        gpu_obf_seeds2 = pi_gen_obf_seed(
            gpu_bi_store2,
            self._gpu_pub_key,
            TEST_SIZE,
            CIPHER_BITS // 6,
            1,
            None)
        cpu_obf_seeds2 = cpu_pi_gen_obf_seed(
            None, self._pub_key, TEST_SIZE, CIPHER_BITS // 6, 1, None
        )
        # obfuscate the other array
        gpu_obfuscated2 = pi_obfuscate(
            self._gpu_pub_key,
            gpu_encrypted2,
            gpu_obf_seeds2,
            gpu_pi_store2,
            None)
        cpu_obfuscated2 = cpu_pi_obfuscate(
            self._pub_key,
            cpu_encrypted2,
            cpu_obf_seeds2,
            [v.exponent for v in cpu_encoded2],
            None,
            None,
        )
        # check intermediate result
        assert_ndarray_diff(
            np.asarray(pi_c2p(gpu_obfuscated2)[0]),
            np.asarray([v.ciphertext(False) for v in cpu_obfuscated2]),
        )

        print("\n>>>>> fp_mul profiling begins")
        gpu_fp_mul_store = fp_alloc(None, TEST_SIZE, MEM_HOST)
        (gpu_fp_mul_res, _), gpu_fp_mul_time = profile(fp_mul)(
            gpu_encoded,
            gpu_encoded2,
            shape_store,
            shape_store,
            gpu_fp_mul_store,
            shape_store,
            None,
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_fp_mul_time
            )
        )
        cpu_fp_mul_res, cpu_fp_mul_time = profile(
            cpu_fp_mul)(cpu_encoded, cpu_encoded2)
        compare_time(gpu_fp_mul_time, cpu_fp_mul_time)
        add_to_perf_reports(
            self._perf_reports,
            "Fixed-point Number Multiply",
            gpu_fp_mul_time,
            cpu_fp_mul_time,
            DATA_SIZE * 2,
        )

        # Compare results
        received_fp_mul_res = fp_c2p(gpu_fp_mul_res)
        for i in range(TEST_SIZE):
            assert_diff(
                received_fp_mul_res[i].encoding,
                cpu_fp_mul_res[i].encoding)
            assert received_fp_mul_res[i].BASE == cpu_fp_mul_res[i].BASE
            assert received_fp_mul_res[i].exponent == cpu_fp_mul_res[i].exponent

        print("\n>>>>> pi_add profiling begins")
        (gpu_add_res, _), gpu_add_time = profile(pi_add)(
            self._gpu_pub_key,
            gpu_obfuscated,
            gpu_obfuscated2,
            shape_store,
            shape_store,
            gpu_pi_store,
            shape_store,
            None,
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_add_time
            )
        )
        cpu_add_res, cpu_add_time = profile(
            lambda a, b: [a[i] + b[i] for i in range(TEST_SIZE)]
        )(cpu_obfuscated, cpu_obfuscated2)
        compare_time(gpu_add_time, cpu_add_time)
        add_to_perf_reports(
            self._perf_reports,
            "Add",
            gpu_add_time,
            cpu_add_time,
            DATA_SIZE * 2)

        print("\n>>>>> pi_mul profiling begins")
        (gpu_mul_res, _), gpu_mul_time = profile(pi_mul)(
            self._gpu_pub_key,
            gpu_add_res,
            gpu_encoded2,
            shape_store,
            shape_store,
            gpu_pi_store,
            shape_store,
            None,
        )
        print(
            "GPU computation completed in {} second(s), waiting for CPU".format(
                gpu_mul_time
            )
        )
        cpu_mul_res, cpu_mul_time = profile(
            lambda a, b: [a[i] * b[i] for i in range(TEST_SIZE)]
        )(cpu_add_res, cpu_encoded2)
        compare_time(gpu_mul_time, cpu_mul_time)
        add_to_perf_reports(
            self._perf_reports,
            "Multiply",
            gpu_mul_time,
            cpu_mul_time,
            DATA_SIZE * 2)

        gpu_pi_matmul_store = pi_alloc(None, NUM_ROWS * NUM_ROWS, MEM_HOST)
        gpu_matmul_res, gpu_matmul_shape = gpu_mul_res, shape_store
        cpu_matmul_res = np.asarray(cpu_mul_res).reshape(shape_tuple)

        print("\n>>>>> pi_sum profiling begins")
        print("shape is", gpu_matmul_shape.to_tuple())
        gpu_pi_sum_store = pi_alloc(None, max(NUM_ROWS, NUM_COLS), MEM_HOST)
        for axis in [0, 1, None]:
            print(">>> axis:", axis)
            (gpu_sum_res, _), gpu_sum_time = profile(pi_sum)(
                self._gpu_pub_key,
                gpu_matmul_res,
                gpu_matmul_shape,
                axis,
                gpu_pi_sum_store,
                None,
                None,
            )
            print(
                "GPU computation completed in {} second(s), waiting for CPU".format(
                    gpu_sum_time
                )
            )
            cpu_sum_res, cpu_sum_time = profile(lambda a: np.sum(a, axis))(
                cpu_matmul_res
            )
            compare_time(gpu_sum_time, cpu_sum_time)
            add_to_perf_reports(
                self._perf_reports,
                "Sum (axis={})".format(axis),
                gpu_sum_time,
                cpu_sum_time,
                DATA_SIZE,
            )

            # check result
            gpu_decrypted = te_c2p(
                pi_decrypt(
                    self._gpu_pub_key,
                    self._gpu_priv_key,
                    gpu_sum_res,
                    None,
                    None,
                    None))
            cpu_decrypted = np.asarray(
                [self._priv_key.decrypt(v) for v in cpu_sum_res.flat]
                if axis is not None
                else [self._priv_key.decrypt(cpu_sum_res)]
            )
            assert_ndarray_diff(gpu_decrypted, cpu_decrypted)

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

    @classmethod
    def tearDownClass(cls):
        for v in cls._perf_reports:
            v.dump_result()


if __name__ == "__main__":
    unittest.main()
