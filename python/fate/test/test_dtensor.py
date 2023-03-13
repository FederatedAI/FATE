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
def t1_i32_sharding():
    return [
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
    ]


@fixture
def t1_f32_sharding():
    return [
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    ]


@fixture
def t2_f32_sharding():
    return [
        torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.float32),
        torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.float32),
        torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.float32),
    ]


@fixture
def t1_i32(ctx, t1_i32_sharding):
    return DTensor.from_sharding_list(
        ctx,
        t1_i32_sharding,
        num_partitions=3,
    )


@fixture
def t1_f32(ctx, t1_f32_sharding):
    return DTensor.from_sharding_list(
        ctx,
        t1_f32_sharding,
        num_partitions=3,
    )


@fixture
def t2_f32(ctx, t2_f32_sharding):
    return DTensor.from_sharding_list(
        ctx,
        t2_f32_sharding,
        num_partitions=3,
    )


@pytest.mark.parametrize(
    "op",
    [torch.exp, torch.log, torch.square],
)
def test_unary(ctx, t1_f32, t1_f32_sharding, op):
    assert op(t1_f32) == DTensor.from_sharding_list(ctx, [op(s) for s in t1_f32_sharding], num_partitions=3)


@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div, torch.rsub],
)
def test_binary(ctx, t1_f32, t2_f32, t1_f32_sharding, t2_f32_sharding, op):
    assert op(t1_f32, t2_f32) == DTensor.from_sharding_list(
        ctx, [op(s1, s2) for s1, s2 in zip(t1_f32_sharding, t2_f32_sharding)], num_partitions=3
    )
