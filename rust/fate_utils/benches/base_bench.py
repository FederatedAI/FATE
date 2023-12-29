import pytest
import operator
import os

import numpy as np
import phe

try:
    import gmpy2
except Exception:
    raise RuntimeError(f"gmpy2 not installed, lib phe without gmpy2 is slow")


def get_num_threads():
    num = int(os.environ.get("NUM_THREADS", 4))
    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count < num:
        raise RuntimeError(
            f"num threads {num} larger than num cpu core deteacted, try specify num threads by `NUM_THREADS=xxx pytest ...`"
        )
    return num


def get_single_thread_keygen():
    from rust_paillier import keygen

    return keygen


NUM_THREADS = get_num_threads()


def get_multiple_thread_keygen():
    from rust_paillier.par import keygen, set_num_threads

    set_num_threads(NUM_THREADS)
    return keygen


# modify this if you want to benchmark your custom packages
BENCH_PACKAGES = {
    "cpu_thread[1]": get_single_thread_keygen(),
    f"cpu_multiple_thread[{NUM_THREADS}]": get_multiple_thread_keygen(),
}

sa, sb, sc, sd = ((11, 21), (11, 21), (21, 11), 21)
a = np.random.random(size=sa).astype(dtype=np.float64) - 0.5
b = np.random.random(size=sb).astype(dtype=np.float64) - 0.5
c = np.random.random(size=sc).astype(dtype=np.float64) - 0.5
d = np.random.random(size=sd).astype(dtype=np.float64) - 0.5


class BaselineSuite:
    def __init__(self, a, b, c, d) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.pk, self.sk = phe.generate_paillier_keypair(n_length=2048)
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


class BenchSuite:
    def __init__(self, a, b, c, d, keygen) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.pk, self.sk = keygen(2048)
        self.ea = self.pk.encrypt_f64(self.a)
        self.eb = self.pk.encrypt_f64(self.b)
        self.ec = self.pk.encrypt_f64(self.c)
        self.ed = self.pk.encrypt_f64(self.d)

    def get(self, name):
        return getattr(self, name)

    def encrypt(self):
        getattr(self.pk, "encrypt_f64")(self.a)

    def decrypt(self):
        getattr(self.sk, "decrypt_f64")(self.ea)

    def add_cipher(self):
        getattr(self.ea, "add_cipherblock")(self.eb)

    def sub_cipher(self):
        getattr(self.ea, "sub_cipherblock")(self.eb)

    def add_plain(self):
        getattr(self.ea, "add_plaintext_f64")(self.b)

    def sub_plain(self):
        getattr(self.ea, "sub_plaintext_f64")(self.b)

    def mul_plain(self):
        getattr(self.ea, "mul_plaintext_f64")(self.b)

    def matmul_plain_ix2(self):
        getattr(self.ea, "matmul_plaintext_ix2_f64")(self.c)

    def rmatmul_plain_ix2(self):
        getattr(self.ec, "rmatmul_plaintext_ix2_f64")(self.a)

    def matmul_plain_ix1(self):
        getattr(self.ea, "matmul_plaintext_ix1_f64")(self.d)

    def rmatmul_plain_ix1(self):
        getattr(self.ec, "rmatmul_plaintext_ix1_f64")(self.d)


def get_suites():
    ids = []
    suites = []
    ids.append("baseline")
    suites.append(BaselineSuite(a, b, c, d))
    for package_id, keygen in BENCH_PACKAGES.items():
        ids.append(package_id)
        suites.append(BenchSuite(a, b, c, d, keygen))
    return ids, suites


ids, suites = get_suites()


def pytest_generate_tests(metafunc):
    if "suite" in metafunc.fixturenames:
        metafunc.parametrize("suite", suites, ids=ids)


def create_tests(func_name):
    @pytest.mark.benchmark(group=f"{func_name}")
    def f(suite, benchmark):
        benchmark(suite.get(func_name))

    f.__name__ = f"test_{func_name}"
    return f


def get_exec_expression(name, *args):
    shapes = []
    for shape in args:
        if isinstance(shape, int):
            shapes.append(f"{shape}")
        if isinstance(shape, tuple):
            shapes.append(f"{'x'.join(map(str, shape))}")
    shape_suffix = "_".join(shapes)
    return f'test_{name}_{shape_suffix} = create_tests("{name}")'


for name, *shapes in [
    ("encrypt", sa),
    ("decrypt", sa),
    ("add_cipher", sa, sb),
    ("sub_cipher", sa, sb),
    ("add_plain", sa, sb),
    ("sub_plain", sa, sb),
    ("mul_plain", sa, sb),
    ("matmul_plain_ix2", sa, sc),
    ("rmatmul_plain_ix2", sc, sa),
    ("matmul_plain_ix1", sa, sd),
    ("rmatmul_plain_ix1", sc, sd),
]:
    exec(get_exec_expression(name, *shapes))
