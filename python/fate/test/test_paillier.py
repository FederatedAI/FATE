import torch
from fate.arch.tensor.paillier import keygen
from pytest import fixture


@fixture
def keypair():
    return keygen(1024)


@fixture
def pub(keypair):
    return keypair[0]


@fixture
def pri(keypair):
    return keypair[1]


def test_aaa(pub):
    raise ValueError(pub.encrypt(torch.zeros((3, 1)))[2])


# @fixture
# def x3_4_p():
#     return torch.randn((3, 4))
#
#
# @fixture
# def x3_4_e(x3_4_p, pub):
#     return pub.encrypt(x3_4_p)
#
#
# @fixture
# def x3_4_p2():
#     return torch.randn((3, 4))
#
#
# @fixture
# def x4_p():
#     return torch.randn((4,))
#
#
# @fixture
# def x4_e(x4_p, pub):
#     return pub.encrypt(x4_p)
#
#
# @fixture
# def x3_p():
#     return torch.randn((3,))
#
#
# @fixture
# def x4_3_p():
#     return torch.randn((4, 3))
#
#
# @fixture
# def x4_3_e(x4_3_p, pub):
#     return pub.encrypt(x4_3_p)
#
#
# def test_add(x3_4_p, x3_4_p2, x3_4_e, pri):
#     # cipher + cipher
#     assert torch.allclose(pri.decrypt(torch.add(x3_4_e, x3_4_e)), torch.add(x3_4_p, x3_4_p))
#
#     assert torch.allclose(pri.decrypt(torch.add(x3_4_e, x3_4_p2)), torch.add(x3_4_p, x3_4_p2))
#     assert torch.allclose(pri.decrypt(x3_4_e + x3_4_p2), x3_4_p + x3_4_p2)
#     assert torch.allclose(pri.decrypt(torch.add(x3_4_p2, x3_4_e)), torch.add(x3_4_p2, x3_4_p))
#     assert torch.allclose(pri.decrypt(x3_4_p2 + x3_4_e), x3_4_p2 + x3_4_p)
#
#
# def test_sub(x3_4_p, x3_4_p2, x3_4_e, pri):
#     # cipher + cipher
#     assert torch.allclose(pri.decrypt(torch.sub(x3_4_e, x3_4_e)), torch.sub(x3_4_p, x3_4_p))
#
#     assert torch.allclose(pri.decrypt(torch.sub(x3_4_e, x3_4_p2)), torch.sub(x3_4_p, x3_4_p2))
#     assert torch.allclose(pri.decrypt(torch.rsub(x3_4_e, x3_4_p2)), torch.rsub(x3_4_p, x3_4_p2))
#     assert torch.allclose(pri.decrypt(x3_4_e - x3_4_p2), x3_4_p - x3_4_p2)
#     assert torch.allclose(pri.decrypt(torch.sub(x3_4_p2, x3_4_e)), torch.sub(x3_4_p2, x3_4_p))
#     assert torch.allclose(pri.decrypt(torch.rsub(x3_4_p2, x3_4_e)), torch.rsub(x3_4_p2, x3_4_p))
#     assert torch.allclose(pri.decrypt(x3_4_p2 - x3_4_e), x3_4_p2 - x3_4_p)
#
#
# def test_mul(x3_4_p, x3_4_p2, x3_4_e, pri):
#     assert torch.allclose(pri.decrypt(torch.mul(x3_4_e, x3_4_p2)), torch.mul(x3_4_p, x3_4_p2))
#     assert torch.allclose(pri.decrypt(x3_4_e * x3_4_p2), x3_4_p * x3_4_p2)
#     assert torch.allclose(pri.decrypt(torch.mul(x3_4_p2, x3_4_e)), torch.mul(x3_4_p2, x3_4_p))
#     assert torch.allclose(pri.decrypt(x3_4_p2 * x3_4_e), x3_4_p2 * x3_4_p)
#
#
# def test_matmul_ix2(x3_4_p, x4_3_p, x3_4_e, x4_e, x4_p, pri):
#     assert torch.allclose(pri.decrypt(torch.matmul(x3_4_e, x4_3_p)), torch.matmul(x3_4_p, x4_3_p))
#     assert torch.allclose(pri.decrypt(torch.matmul(x4_3_p, x3_4_e)), torch.matmul(x4_3_p, x3_4_p))
#     assert torch.allclose(pri.decrypt(torch.rmatmul_f(x3_4_e, x4_3_p)), torch.rmatmul_f(x3_4_p, x4_3_p))
#     assert torch.allclose(pri.decrypt(torch.rmatmul_f(x4_3_p, x3_4_e)), torch.rmatmul_f(x4_3_p, x3_4_p))
#     assert torch.allclose(pri.decrypt(x3_4_e @ x4_3_p), x3_4_p @ x4_3_p)
#     assert torch.allclose(pri.decrypt(x4_3_p @ x3_4_e), x4_3_p @ x3_4_p)
#
#     assert torch.allclose(pri.decrypt(torch.matmul(x4_e, x4_3_p)), torch.matmul(x4_p, x4_3_p))
#     assert torch.allclose(pri.decrypt(torch.matmul(x3_4_p, x4_e)), torch.matmul(x3_4_p, x4_p))
#     assert torch.allclose(pri.decrypt(torch.rmatmul_f(x4_e, x3_4_p)), torch.rmatmul_f(x4_p, x3_4_p))
#
#
# def test_matmul_ix1(x3_4_p, x3_4_e, x4_p, x4_e, x3_p, pri):
#     assert torch.allclose(pri.decrypt(torch.matmul(x3_4_e, x4_p)), torch.matmul(x3_4_p, x4_p))
#     assert torch.allclose(pri.decrypt(x3_4_e @ x4_p), x3_4_p @ x4_p)
#     assert torch.allclose(pri.decrypt(torch.rmatmul_f(x3_4_e, x3_p)), torch.rmatmul_f(x3_4_p, x3_p))
#     assert torch.allclose(pri.decrypt(x3_p @ x3_4_e), x3_p @ x3_4_p)
#
#     assert torch.allclose(pri.decrypt(torch.matmul(x4_e, x4_p)), torch.matmul(x4_p, x4_p))
