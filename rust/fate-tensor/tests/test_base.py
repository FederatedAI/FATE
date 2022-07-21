import operator

import fate_tensor
import numpy as np
import pytest
import cachetools

pk, sk = fate_tensor.keygen(1024)


def encrypt(fp, par, data):
    if par:
        return getattr(pk, f"encrypt_{fp}_par")(data)
    else:
        return getattr(pk, f"encrypt_{fp}")(data)


def decrypt(fp, par, data):
    if par:
        return getattr(sk, f"decrypt_{fp}_par")(data)
    else:
        return getattr(sk, f"decrypt_{fp}")(data)


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


def test_keygen():
    fate_tensor.keygen(1024)


def cipher_op(ciphertext, op, par):
    if par:
        return getattr(ciphertext, f"{op.__name__}_cipherblock_par")
    else:
        return getattr(ciphertext, f"{op.__name__}_cipherblock")


def plaintest_op(ciphertext, op, par, fp, scalar=False):
    if par:
        if scalar:
            return getattr(ciphertext, f"{op.__name__}_plaintext_scalar_{fp}_par")
        else:
            return getattr(ciphertext, f"{op.__name__}_plaintext_{fp}_par")
    else:
        if scalar:
            return getattr(ciphertext, f"{op.__name__}_plaintext_scalar_{fp}")
        else:
            return getattr(ciphertext, f"{op.__name__}_plaintext_{fp}")


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i32", "i64"])
def test_cipher(par, fp):
    e = decrypt(fp, par, encrypt(fp, par, data(fp, 0)))
    c = data(fp, 0)
    assert np.isclose(e, c).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_cipher_op(par, fp, op):
    ea = encrypt(fp, par, data(fp, 0))
    eb = encrypt(fp, par, data(fp, 1))
    result = cipher_op(ea, op, par)(eb)
    expect = op(data(fp, 0), data(fp, 1))
    diff = decrypt(fp, par, result) - expect
    assert np.isclose(diff, 0).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_plaintext_op(par, fp, op):
    ea = encrypt(fp, par, data(fp, 0))
    b = data(fp, 1)
    result = plaintest_op(ea, op, par, fp)(b)
    expect = op(data(fp, 0), data(fp, 1))
    diff = decrypt(fp, par, result) - expect
    assert np.isclose(diff, 0).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_plaintext_op_scalar(par, fp, op):
    ea = encrypt(fp, par, data(fp, 0))
    b = data(fp, 1, scalar=True)
    result = plaintest_op(ea, op, par, fp, True)(b)
    expect = op(data(fp, 0), data(fp, 1, scalar=True))
    diff = decrypt(fp, par, result) - expect
    assert np.isclose(diff, 0).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_matmul_ix2(par, fp):
    a = data(fp, 0, (11, 17))
    b = data(fp, 0, (17, 5))
    ea = encrypt(fp, par, a)
    eab = getattr(ea, f"matmul_plaintext_ix2_{fp}")(b)
    ab = decrypt(fp, par, eab)
    assert np.isclose(ab, a @ b).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_matmul_ix1(par, fp):
    a = data(fp, 0, (11, 17))
    b = data(fp, 0, 17)
    ea = encrypt(fp, par, a)
    eab = getattr(ea, f"matmul_plaintext_ix1_{fp}")(b)
    ab = decrypt(fp, par, eab)
    assert np.isclose(ab, (a @ b).reshape(ab.shape)).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_rmatmul_ix2(par, fp):
    a = data(fp, 0, (11, 17))
    b = data(fp, 0, (17, 5))
    eb = encrypt(fp, par, b)
    reab = getattr(eb, f"rmatmul_plaintext_ix2_{fp}")(a)
    rab = decrypt(fp, par, reab)
    assert np.isclose(rab, a @ b).all()


@pytest.mark.parametrize("par", [False, True])
@pytest.mark.parametrize("fp", ["f64", "f32", "i64", "i32"])
def test_rmatmul_ix1(par, fp):
    a = data(fp, 0, 17)
    b = data(fp, 0, (17, 5))
    eb = encrypt(fp, par, b)
    reab = getattr(eb, f"rmatmul_plaintext_ix1_{fp}")(a)
    rab = decrypt(fp, par, reab)
    assert np.isclose(rab, (a @ b).reshape(rab.shape)).all()
