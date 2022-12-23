import torch
from fate.arch import Context, tensor
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
from pytest import fixture
from pytest_lazyfixture import lazy_fixture


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
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ],
    )


@fixture
def t3(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ],
        d_axis=1,
    )


def test_sum(t1):
    print(tensor.sum(t1))
    print(tensor.sum(t1, dim=0))
    print(tensor.sum(t1, dim=1))
    print(torch.sum(t1.to_local().storage.data))
    print(torch.sum(t1.to_local().storage.data, dim=0))
    print(torch.sum(t1.to_local().storage.data, dim=1))


def test_mean(t1):
    print(tensor.mean(t1))
    print(tensor.mean(t1, dim=0))
    print(tensor.mean(t1, dim=1))
    print(torch.mean(t1.to_local().storage.data))
    print(torch.mean(t1.to_local().storage.data, dim=0))
    print(torch.mean(t1.to_local().storage.data, dim=1))


def test_std(t1):
    print(tensor.std(t1, unbiased=False))
    print(tensor.std(t1, dim=0, unbiased=False))
    print(tensor.std(t1, dim=1, unbiased=False).to_local())
    print(torch.std(t1.to_local().storage.data, unbiased=False))
    print(torch.std(t1.to_local().storage.data, dim=0, unbiased=False))
    print(torch.std(t1.to_local().storage.data, dim=1, unbiased=False))


def test_var(t1):
    print(tensor.var(t1, unbiased=False))
    print(tensor.var(t1, dim=0, unbiased=False))
    print(tensor.var(t1, dim=1, unbiased=False).to_local())
    print(torch.var(t1.to_local().storage.data, unbiased=False))
    print(torch.var(t1.to_local().storage.data, dim=0, unbiased=False))
    print(torch.var(t1.to_local().storage.data, dim=1, unbiased=False))
