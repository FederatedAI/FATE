import torch
from fate.arch import Context, tensor
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
from pytest import fixture


@fixture
def ctx():
    computing = CSession()
    return Context(
        "guest",
        computing=computing,
        federation=StandaloneFederation(computing, "fed", ("guest", 10000), [("host", 9999)]),
    )


@fixture
def t1(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ],
    )


@fixture
def t2(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ],
        d_axis=1,
    )


@fixture
def t3():
    return tensor.tensor(
        torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ),
    )


@fixture
def t4():
    return torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )


def test_sum_all(t1, t2, t3, t4):
    s1 = tensor.sum(t1)
    s2 = tensor.sum(t2)
    s3 = tensor.sum(t3)
    s4 = torch.sum(t4)
    assert s1.to_local().storage.data == s4
    assert s2.to_local().storage.data == s4
    assert s3.to_local().storage.data == s4


def test_sum_dim_0(t1, t2, t3, t4):
    s1 = tensor.sum(t1, dim=0)
    s2 = tensor.sum(t2, dim=0)
    s3 = tensor.sum(t3, dim=0)
    s4 = torch.sum(t4, dim=0)
    s5 = torch.sum(t4.T, dim=0, keepdim=True)
    assert torch.allclose(s1.to_local().storage.data, s4)
    assert torch.allclose(s2.to_local().storage.data, s5)
    assert torch.allclose(s3.to_local().storage.data, s4)


def test_sum_dim_1(t1, t2, t3, t4):
    s1 = tensor.sum(t1, dim=1)
    s2 = tensor.sum(t2, dim=1)
    s3 = tensor.sum(t3, dim=1, keepdim=True)
    s4 = torch.sum(t4, dim=1, keepdim=True)
    s5 = torch.sum(t4.T, dim=1, keepdim=True)
    assert torch.allclose(s1.to_local().storage.data, s4)
    assert torch.allclose(s3.to_local().storage.data, s4)
    assert torch.allclose(s2.to_local().storage.data, s5)


def test_mean(t1, t2):
    print(tensor.mean(t1))
    print(tensor.mean(t1, dim=0))
    print(tensor.mean(t1, dim=1).to_local())
    print("-----------------------------------------------")
    print(tensor.mean(t2))
    print(tensor.mean(t2, dim=0))
    print(tensor.mean(t2, dim=1))
    print("-----------------------------------------------")
    print(torch.mean(t1.to_local().storage.data))
    print(torch.mean(t1.to_local().storage.data, dim=0))
    print(torch.mean(t1.to_local().storage.data, dim=1))


def test_std(t1, t2):
    print(tensor.std(t1, unbiased=False))
    print(tensor.std(t1, dim=0, unbiased=False))
    print(tensor.std(t1, dim=1, unbiased=False).to_local())
    print("-----------------------------------------------")
    print(tensor.std(t2, unbiased=False))
    print(tensor.std(t2, dim=0, unbiased=False))
    print(tensor.std(t2, dim=1, unbiased=False).to_local())
    print("-----------------------------------------------")
    print(torch.std(t1.to_local().storage.data, unbiased=False))
    print(torch.std(t1.to_local().storage.data, dim=0, unbiased=False))
    print(torch.std(t1.to_local().storage.data, dim=1, unbiased=False))


def test_var(t1, t2):
    print(tensor.var(t1, unbiased=False))
    print(tensor.var(t1, dim=0, unbiased=False))
    print(tensor.var(t1, dim=1, unbiased=False).to_local())
    print("-----------------------------------------------")
    print(tensor.var(t2, unbiased=False))
    print(tensor.var(t2, dim=0, unbiased=False))
    print(tensor.var(t2, dim=1, unbiased=False).to_local())
    print("-----------------------------------------------")
    print(torch.var(t1.to_local().storage.data, unbiased=False))
    print(torch.var(t1.to_local().storage.data, dim=0, unbiased=False))
    print(torch.var(t1.to_local().storage.data, dim=1, unbiased=False))


def test_max(t1, t2):
    print(tensor.max(t1))
    print(tensor.max(t1, dim=0))
    print(tensor.max(t1, dim=1).to_local())
    print("-----------------------------------------------")
    print(tensor.max(t2))
    print(tensor.max(t2, dim=0))
    print(tensor.max(t2, dim=1).to_local())
    print("-----------------------------------------------")
    print(torch.max(t1.to_local().storage.data))
    print(torch.max(t1.to_local().storage.data, dim=0).values)
    print(torch.max(t1.to_local().storage.data, dim=1).values)


def test_min(t1, t2):
    print(tensor.min(t1))
    print(tensor.min(t1, dim=0))
    print(tensor.min(t1, dim=1).to_local())
    print("-----------------------------------------------")
    print(tensor.min(t2))
    print(tensor.min(t2, dim=0))
    print(tensor.min(t2, dim=1).to_local())
    print("-----------------------------------------------")
    print(torch.min(t1.to_local().storage.data))
    print(torch.min(t1.to_local().storage.data, dim=0).values)
    print(torch.min(t1.to_local().storage.data, dim=1).values)
