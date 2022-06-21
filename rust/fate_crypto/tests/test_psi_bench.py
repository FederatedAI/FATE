import random
import hashlib
from fate_crypto.psi import Curve25519


def ecdh(k, m):
    return k.encrypt(m)


def dh(k, e):
    return k.diffie_hellman(e)


def sha256(value):
    return hashlib.sha256(bytes(value, encoding="utf-8")).digest()


def test_ecdh_encrypt_bench(benchmark):
    k = Curve25519()
    m = random.SystemRandom().getrandbits(256).to_bytes(32, "little")
    result = benchmark(ecdh, k, m)


def test_ecdh_dh_bench(benchmark):
    k = Curve25519()
    m = random.SystemRandom().getrandbits(256).to_bytes(32, "little")
    e = k.encrypt(m)
    result = benchmark(dh, k, e)


def test_sha256_bench(benchmark):
    m = "1000000000"
    result = benchmark(sha256, m)
