import pytest
import torch
from fate.arch import Backend, Context, tensor
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context, disable_inner_logs
from fate.arch.federation.standalone import StandaloneFederation
from pytest import fixture
from pytest_lazyfixture import lazy_fixture

# disable_inner_logs()


@fixture
def ctx():
    computing = CSession()
    return Context(
        "guest",
        backend=Backend.STANDALONE,
        computing=computing,
        federation=StandaloneFederation(
            computing, "fed", ("guest", 10000), [("host", 9999)]
        ),
    )


@fixture
def t1():
    return tensor.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))


@fixture
def t2():
    return tensor.tensor(torch.tensor([[2, 2, 3], [4, 4, 6]]))


@fixture
def t1_add_t2():
    return tensor.tensor(torch.tensor([[3, 4, 6], [8, 9, 12]]))


@fixture
def t3():
    return tensor.tensor(torch.tensor([[4, 4, 6]]))


@fixture
def dt1(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ],
    )


@fixture
def dt2(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ],
    )


@fixture
def dt1_add_dt2(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[2, 4, 6], [8, 10, 12]]),
            torch.tensor([[2, 4, 6], [8, 10, 12]]),
            torch.tensor([[2, 4, 6], [8, 10, 12]]),
        ],
    )


@fixture
def dt1_add_t3(ctx):
    return tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[5, 6, 9], [8, 9, 12]]),
            torch.tensor([[5, 6, 9], [8, 9, 12]]),
            torch.tensor([[5, 6, 9], [8, 9, 12]]),
        ],
    )


@pytest.mark.parametrize(
    "a,b,c",
    [
        (
            lazy_fixture("t1"),
            lazy_fixture("t2"),
            lazy_fixture("t1_add_t2"),
        ),
        (
            lazy_fixture("dt1"),
            lazy_fixture("dt2"),
            lazy_fixture("dt1_add_dt2"),
        ),
        (lazy_fixture("dt1"), lazy_fixture("t3"), lazy_fixture("dt1_add_t3")),
    ],
)
def test_add(a, b, c):
    assert tensor.add(a, b) == c


def test_exp():
    a = torch.randn((3, 4))
    assert tensor.exp(tensor.tensor(a)) == tensor.tensor(torch.exp(a))
