import operator

import fate_tensor
import numpy as np
import pytest

shape = (2, 2)
testdata = {
    (0, "f64"): np.random.random(shape).astype(np.float64),
    (1, "f64"): np.random.random(shape).astype(np.float64),
    (0, "f32"): np.random.random(shape).astype(np.float32),
    (1, "f32"): np.random.random(shape).astype(np.float32),
    (0, "i64"): np.random.randint(low=2147483648, high=9223372036854775807, size=shape, dtype=np.int64),
    (1, "i64"): np.random.randint(low=-9223372036854775808, high=-2147483648, size=shape, dtype=np.int64),
    (0, "i32"): np.random.randint(low=-100, high=100, size=shape, dtype=np.int32),
    (1, "i32"): np.random.randint(low=-100, high=100, size=shape, dtype=np.int32),
}
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


def data(fp, index):
    return testdata[(index, fp)]


def test_keygen():
    fate_tensor.keygen(1024)


def cipher_op(ciphertext, op, par):
    if par:
        return getattr(ciphertext, f"{op.__name__}_cipherblock_par")
    else:
        return getattr(ciphertext, f"{op.__name__}_cipherblock")


def plaintest_op(ciphertext, op, par, fp):
    if par:
        return getattr(ciphertext, f"{op.__name__}_plaintext_{fp}_par")
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
