import pytest
import torch
from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation
from fate.arch.tensor import DTensor
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
def t1_sharding():
    return [
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
    ]


@fixture
def t1(ctx, t1_sharding):
    return DTensor.from_sharding_list(
        ctx,
        t1_sharding,
        num_partitions=3,
    )


@pytest.mark.parametrize(
    "op",
    [torch.exp, torch.log, torch.square],
)
def test_unary(ctx, t1, t1_sharding, op):
    assert op(t1) == DTensor.from_sharding_list(ctx, [op(s) for s in t1_sharding], num_partitions=3)
