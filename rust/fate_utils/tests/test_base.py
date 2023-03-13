import importlib
import operator

import cachetools
import numpy as np
import pytest
import pickle


def get_suites():
    suites = []
    packages = ["fate_utils.tensor"]
    for package in packages:
        module = importlib.import_module(package)
        suites.append(Suite(module.keygen))
    return suites


class Suite:
    def __init__(self, keygen) -> None:
        self.pk, self.sk = keygen(1024)

    def encrypt(self, fp, data):
        return getattr(self.pk, f"encrypt_{fp}")(data)

    def decrypt(self, fp, data):
        return getattr(self.sk, f"decrypt_{fp}")(data)

    def cipher_op(self, ciphertext, op):
        return getattr(ciphertext, f"{op.__name__}_cipherblock")

    def plaintest_op(self, ciphertext, op, fp, scalar=False):
        if scalar:
            return getattr(ciphertext, f"{op.__name__}_plaintext_scalar_{fp}")
        else:
            return getattr(ciphertext, f"{op.__name__}_plaintext_{fp}")


def pytest_generate_tests(metafunc):
    if "suite" in metafunc.fixturenames:
        metafunc.parametrize("suite", get_suites())


@cachetools.cached({})
def data(fp, index, shape=(3, 5), scalar=False) -> np.ndarray:
    if not scalar:
        if fp == "f64":
            return np.random.random(shape).astype(np.float64) - 0.5
        if fp == "f32":
            return np.random.random(shape).astype(np.float32) - 0.5
        if fp == "i64":
            return np.random.randint(low=2147483648, high=2147483648000, size=shape, dtype=np.int64)
        if fp == "i32":
            return np.random.randint(low=-100, high=100, size=shape, dtype=np.int32)
    else:
        if fp == "f64":
            return np.random.random(1).astype(np.float64)[0] - 0.5
        if fp == "f32":
            return np.random.random(1).astype(np.float32)[0] - 0.5
        if fp == "i64":
            return np.random.randint(low=2147483648, high=2147483648000, size=1, dtype=np.int64)[0]
        if fp == "i32":
            return np.random.randint(low=-100, high=100, size=1, dtype=np.int32)[0]


@pytest.mark.parametrize("fp", ["f64"])
def test_serde(suite: Suite, fp):
    assert suite.pk == pickle.loads(pickle.dumps(suite.pk))
    assert suite.sk == pickle.loads(pickle.dumps(suite.sk))


@pytest.mark.parametrize("fp", ["f64", "f32", "i32", "i64"])
def test_cipher(suite: Suite, fp):
    e = suite.decrypt(fp, suite.encrypt(fp, data(fp, 0)))
    c = data(fp, 0)
    assert np.isclose(e, c).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_cipher_op(suite, fp, op):
    ea = suite.encrypt(fp, data(fp, 0))
    eb = suite.encrypt(fp, data(fp, 1))
    result = suite.cipher_op(ea, op)(eb)
    expect = op(data(fp, 0), data(fp, 1))
    diff = suite.decrypt(fp, result) - expect
    assert np.isclose(diff, 0).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_plaintext_op(suite: Suite, fp, op):
    ea = suite.encrypt(fp, data(fp, 0))
    b = data(fp, 1)
    result = suite.plaintest_op(ea, op, fp)(b)
    expect = op(data(fp, 0), data(fp, 1))
    diff = suite.decrypt(fp, result) - expect
    assert np.isclose(diff, 0).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_plaintext_op_scalar(suite: Suite, fp, op):
    ea = suite.encrypt(fp, data(fp, 0))
    b = data(fp, 1, scalar=True)
    result = suite.plaintest_op(ea, op, fp, True)(b)
    expect = op(data(fp, 0), data(fp, 1, scalar=True))
    diff = suite.decrypt(fp, result) - expect
    assert np.isclose(diff, 0).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_matmul_ix2(suite: Suite, fp):
    a = data(fp, 0, (11, 17))
    b = data(fp, 0, (17, 5))
    ea = suite.encrypt(fp, a)
    eab = getattr(ea, f"matmul_plaintext_ix2_{fp}")(b)
    ab = suite.decrypt(fp, eab)
    assert np.isclose(ab, a @ b).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_matmul_ix1(suite: Suite, fp):
    a = data(fp, 0, (11, 17))
    b = data(fp, 0, 17)
    ea = suite.encrypt(fp, a)
    eab = getattr(ea, f"matmul_plaintext_ix1_{fp}")(b)
    ab = suite.decrypt(fp, eab)
    assert np.isclose(ab, (a @ b).reshape(ab.shape)).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_rmatmul_ix2(suite: Suite, fp):
    a = data(fp, 0, (11, 17))
    b = data(fp, 0, (17, 5))
    eb = suite.encrypt(fp, b)
    reab = getattr(eb, f"rmatmul_plaintext_ix2_{fp}")(a)
    rab = suite.decrypt(fp, reab)
    assert np.isclose(rab, a @ b).all()


@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_rmatmul_ix1(suite: Suite, fp):
    a = data(fp, 0, 17)
    b = data(fp, 0, (17, 5))
    eb = suite.encrypt(fp, b)
    reab = getattr(eb, f"rmatmul_plaintext_ix1_{fp}")(a)
    rab = suite.decrypt(fp, reab)
    assert np.isclose(rab, (a @ b).reshape(rab.shape)).all()
