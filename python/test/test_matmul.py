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
def t3():
    return tensor.tensor(
        torch.tensor([[1.0], [1.0], [1.0]]),
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
def t4():
    return torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )


def test_1(t1, t3):
    print(t1)
    print(t3)
    print(tensor.matmul(t1, t3))
