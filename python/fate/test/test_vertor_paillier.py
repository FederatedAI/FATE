import torch
from fate.arch.tensor._cipher import phe_keygen

kit = phe_keygen("paillier_vector_based", {"key_length": 1024})
pk, sk, coder = kit.pk, kit.sk, kit.coder


def test_add():
    encoded = coder.encode(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    encrypted = pk.encrypt(encoded)
    double_encrypted = torch.add(encrypted, encrypted)
    double_encrypted = torch.add(double_encrypted, 1)
    double_encrypted = torch.add(double_encrypted, torch.rand(2, 4))
    double_encrypted = torch.add(double_encrypted, torch.tensor(0.3))
    decrypted = sk.decrypt(double_encrypted)
    decoded = coder.decode(decrypted)
    print(decoded)


def test_sub():
    encoded = coder.encode(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    encrypted = pk.encrypt(encoded)
    double_encrypted = torch.sub(encrypted, encrypted)
    double_encrypted = torch.sub(double_encrypted, 1)
    double_encrypted = torch.sub(double_encrypted, torch.rand(2, 4))
    double_encrypted = torch.sub(double_encrypted, torch.tensor(0.3))
    decrypted = sk.decrypt(double_encrypted)
    decoded = coder.decode(decrypted)
    print(decoded)


def test_rsub():
    encoded = coder.encode(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    encrypted = pk.encrypt(encoded)
    double_encrypted = torch.rsub(encrypted, encrypted)
    double_encrypted = torch.rsub(double_encrypted, 1)
    double_encrypted = torch.rsub(double_encrypted, torch.rand(2, 4))
    double_encrypted = torch.rsub(double_encrypted, torch.tensor(0.3))
    decrypted = sk.decrypt(double_encrypted)
    decoded = coder.decode(decrypted)
    print(decoded)


def test_mul():
    encoded = coder.encode(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    encrypted = pk.encrypt(encoded)
    double_encrypted = torch.mul(encrypted, 2)
    double_encrypted = torch.mul(double_encrypted, torch.rand(2, 4))
    decrypted = sk.decrypt(double_encrypted)
    decoded = coder.decode(decrypted)
    print(decoded)


def test_matmul():
    x = torch.rand(5, 2)
    y = torch.rand(2, 4)
    enc_x = pk.encrypt(coder.encode(x))
    enc_z = torch.matmul(enc_x, y)
    z = coder.decode(sk.decrypt(enc_z))
    assert torch.allclose(z, torch.matmul(x, y))


def test_rmatmul():
    x = torch.rand(2, 5)
    y = torch.rand(4, 2)
    enc_x = pk.encrypt(coder.encode(x))
    enc_z = torch.rmatmul_f(enc_x, y)
    z = coder.decode(sk.decrypt(enc_z))
    assert torch.allclose(z, torch.matmul(y, x))
