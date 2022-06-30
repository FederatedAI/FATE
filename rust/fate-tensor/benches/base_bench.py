import fate_tensor
import numpy as np
import pytest
import operator
import phe


class PHESuite:
    def __init__(self, a, b, c, d) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.pk, self.sk = phe.generate_paillier_keypair(n_length=1024)
        self.ea = np.vectorize(self.pk.encrypt)(self.a)
        self.eb = np.vectorize(self.pk.encrypt)(self.b)
        self.ec = np.vectorize(self.pk.encrypt)(self.c)
        self.ed = np.vectorize(self.pk.encrypt)(self.d)

    def get(self, name):
        return getattr(self, name)

    def encrypt(self):
        return np.vectorize(self.pk.encrypt)(self.a)

    def decrypt(self):
        return np.vectorize(self.sk.decrypt)(self.ea)

    def add_cipher(self):
        np.vectorize(operator.add)(self.ea, self.eb)

    def sub_cipher(self):
        np.vectorize(operator.sub)(self.ea, self.eb)

    def add_plain(self):
        return np.vectorize(operator.add)(self.ea, self.b)

    def sub_plain(self):
        return np.vectorize(operator.sub)(self.ea, self.b)

    def mul_plain(self):
        return np.vectorize(operator.mul)(self.ea, self.b)

    def matmul_plain_ix2(self):
        return self.ea @ self.c

    def rmatmul_plain_ix2(self):
        return self.a @ self.ec

    def matmul_plain_ix1(self):
        return self.ea @ self.d

    def rmatmul_plain_ix1(self):
        return self.d @ self.ec


class CPUBlockSuite:
    _mix = ""

    def __init__(self, a, b, c, d) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.pk, self.sk = fate_tensor.keygen(1024)
        self.ea = self.pk.encrypt_f64(self.a)
        self.eb = self.pk.encrypt_f64(self.b)
        self.ec = self.pk.encrypt_f64(self.c)
        self.ed = self.pk.encrypt_f64(self.d)

    def mix(self, name):
        return f"{name}{self._mix}"

    def get(self, name):
        return getattr(self, name)

    def encrypt(self):
        getattr(self.pk, self.mix("encrypt_f64"))(self.a)

    def decrypt(self):
        getattr(self.sk, self.mix("decrypt_f64"))(self.ea)

    def add_cipher(self):
        getattr(self.ea, self.mix("add_cipherblock"))(self.eb)

    def sub_cipher(self):
        getattr(self.ea, self.mix("sub_cipherblock"))(self.eb)

    def add_plain(self):
        getattr(self.ea, self.mix("add_plaintext_f64"))(self.b)

    def sub_plain(self):
        getattr(self.ea, self.mix("sub_plaintext_f64"))(self.b)

    def mul_plain(self):
        getattr(self.ea, self.mix("mul_plaintext_f64"))(self.b)

    def matmul_plain_ix2(self):
        getattr(self.ea, self.mix("matmul_plaintext_ix2_f64"))(self.c)

    def rmatmul_plain_ix2(self):
        getattr(self.ec, self.mix("rmatmul_plaintext_ix2_f64"))(self.a)

    def matmul_plain_ix1(self):
        getattr(self.ea, self.mix("matmul_plaintext_ix1_f64"))(self.d)

    def rmatmul_plain_ix1(self):
        getattr(self.ec, self.mix("rmatmul_plaintext_ix1_f64"))(self.d)


class CPUBlockParSuite(CPUBlockSuite):
    _mix = "_par"


class Suites:
    def __init__(self, a, b, c, d) -> None:
        self.suites = {
            "phe": PHESuite(a, b, c, d),
            "block": CPUBlockSuite(a, b, c, d),
            "block_par": CPUBlockParSuite(a, b, c, d),
        }

    def get(self, name):
        return self.suites[name]


@pytest.fixture
def shape():
    return ((11, 21), (11, 21), (21, 11), 21)


@pytest.fixture
def suites(shape):
    sa, sb, sc, sd = shape
    a = np.random.random(size=sa).astype(dtype=np.float64) - 0.5
    b = np.random.random(size=sb).astype(dtype=np.float64) - 0.5
    c = np.random.random(size=sc).astype(dtype=np.float64) - 0.5
    d = np.random.random(size=sd).astype(dtype=np.float64) - 0.5
    return Suites(a, b, c, d)


def create_tests(func_name):
    # @pytest.mark.benchmark(group=func_name)
    @pytest.mark.parametrize("name", ["phe", "block", "block_par"])
    def f(name, suites, benchmark):
        benchmark(suites.get(name).get(func_name))

    f.__name__ = f"test_{func_name}"
    return f


test_encrypt = create_tests("encrypt")
test_decrypt = create_tests("decrypt")
test_add_cipher = create_tests("add_cipher")
test_sub_cipher = create_tests("sub_cipher")
test_add_plain = create_tests("add_plain")
test_sub_plain = create_tests("sub_plain")
test_mul_plain = create_tests("mul_plain")
test_matmul_plain_ix2 = create_tests("matmul_plain_ix2")
test_rmatmul_plain_ix2 = create_tests("rmatmul_plain_ix2")
test_matmul_plain_ix1 = create_tests("matmul_plain_ix1")
test_rmatmul_plain_ix1 = create_tests("rmatmul_plain_ix1")
