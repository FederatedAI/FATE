import torch
from fate.arch import Context

ctx = Context()
kit = ctx.cipher.phe.setup({"kind": "paillier", "key_length": 1024})
pk = kit.get_tensor_encryptor()
sk = kit.get_tensor_decryptor()


def test_add():
    encrypted = pk.encrypt_tensor(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    double_encrypted = torch.add(encrypted, encrypted)
    double_encrypted = torch.add(double_encrypted, 1)
    double_encrypted = torch.add(double_encrypted, torch.rand(2, 4))
    double_encrypted = torch.add(double_encrypted, torch.tensor(0.3))
    decrypted = sk.decrypt_tensor(double_encrypted)
    print(decrypted)


def test_sub():
    encrypted = pk.encrypt_tensor(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    double_encrypted = torch.sub(encrypted, encrypted)
    double_encrypted = torch.sub(double_encrypted, 1)
    double_encrypted = torch.sub(double_encrypted, torch.rand(2, 4))
    double_encrypted = torch.sub(double_encrypted, torch.tensor(0.3))
    decrypted = sk.decrypt_tensor(double_encrypted)
    print(decrypted)


def test_rsub():
    encrypted = pk.encrypt_tensor(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    double_encrypted = torch.rsub(encrypted, encrypted)
    double_encrypted = torch.rsub(double_encrypted, 1)
    double_encrypted = torch.rsub(double_encrypted, torch.rand(2, 4))
    double_encrypted = torch.rsub(double_encrypted, torch.tensor(0.3))
    decrypted = sk.decrypt_tensor(double_encrypted)
    print(decrypted)


def test_mul():
    encrypted = pk.encrypt_tensor(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, -8.0]]))
    double_encrypted = torch.mul(encrypted, 2)
    double_encrypted = torch.mul(double_encrypted, torch.rand(2, 4))
    decrypted = sk.decrypt_tensor(double_encrypted)
    print(decrypted)


def test_matmul():
    x = torch.rand(5, 2)
    y = torch.rand(2, 4)
    enc_x = pk.encrypt_tensor(x)
    enc_z = torch.matmul(enc_x, y)
    z = sk.decrypt_tensor(enc_z)
    assert torch.allclose(z, torch.matmul(x, y))


def test_rmatmul():
    x = torch.rand(2, 5)
    y = torch.rand(4, 2)
    enc_x = pk.encrypt_tensor(x)
    enc_z = torch.rmatmul_f(enc_x, y)
    z = sk.decrypt_tensor(enc_z)
    assert torch.allclose(z, torch.matmul(y, x))
