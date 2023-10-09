import pytest
import torch
from fate.arch import Context
from fate.arch.computing.standalone import CSession
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


def test_cipher(ctx, t1_f32):
    kit = ctx.cipher.phe.setup({})
    encryptor, decryptor = kit.get_tensor_encryptor(), kit.get_tensor_decryptor()
    encrypted = encryptor.encrypt_tensor(t1_f32)
    print(torch.to_local_f(decryptor.decrypt_tensor(encrypted)))


@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div, torch.rsub],
)
def test_binary(ctx, t1_f32, t2_f32, t1_f32_sharding, t2_f32_sharding, op):
    assert op(t1_f32, t2_f32) == DTensor.from_sharding_list(
        ctx, [op(s1, s2) for s1, s2 in zip(t1_f32_sharding, t2_f32_sharding)], num_partitions=3
    )


@pytest.mark.parametrize(
    "op",
    [torch.sum, torch.mean],
)
def test_sum_mean(ctx, t1_f32, t2_f32, t1_f32_sharding, t2_f32_sharding, op):
    assert op(t1_f32) == op(torch.cat(t1_f32_sharding))
    assert torch.allclose(op(t1_f32, dim=0), op(torch.cat(t1_f32_sharding), dim=0))
    assert op(t1_f32, dim=1) == DTensor.from_sharding_list(
        ctx, [op(s, dim=1) for s in t1_f32_sharding], num_partitions=3
    )
    assert op(t1_f32, dim=1, keepdim=True) == DTensor.from_sharding_list(
        ctx, [op(s, dim=1, keepdim=True) for s in t1_f32_sharding], num_partitions=3
    )


@pytest.mark.parametrize(
    "op",
    [torch.var, torch.std],
)
def test_var_std(ctx, t1_f32, t2_f32, t1_f32_sharding, t2_f32_sharding, op):
    assert torch.isclose(op(t1_f32), op(torch.cat(t1_f32_sharding)))
    assert torch.allclose(op(t1_f32, dim=0), op(torch.cat(t1_f32_sharding), dim=0))
    assert torch.allclose(op(t1_f32, dim=0, unbiased=False), op(torch.cat(t1_f32_sharding), dim=0, unbiased=False))
    assert op(t1_f32, dim=1) == DTensor.from_sharding_list(
        ctx, [op(s, dim=1) for s in t1_f32_sharding], num_partitions=3
    )
    assert op(t1_f32, dim=1, keepdim=True) == DTensor.from_sharding_list(
        ctx, [op(s, dim=1, keepdim=True) for s in t1_f32_sharding], num_partitions=3
    )


@pytest.mark.parametrize(
    "op",
    [torch.max, torch.min],
)
def test_max_min(ctx, t1_f32, t2_f32, t1_f32_sharding, t2_f32_sharding, op):
    assert torch.isclose(op(t1_f32), op(torch.cat(t1_f32_sharding)))

    def _eq(r1, r2):
        assert r1.indices.shape == r2.indices.shape
        assert r1.values.shape == r2.values.shape
        assert torch.allclose(r1.indices, r2.indices)
        assert torch.allclose(r1.values, r2.values)

    _eq(op(t1_f32, dim=0), op(torch.cat(t1_f32_sharding), dim=0))
    _eq(op(t1_f32, dim=0, keepdim=True), op(torch.cat(t1_f32_sharding), dim=0, keepdim=True))

    assert op(t1_f32, dim=1).values == DTensor.from_sharding_list(
        ctx, [op(s, dim=1).values for s in t1_f32_sharding], num_partitions=3
    )

    assert op(t1_f32, dim=1, keepdim=True).values == DTensor.from_sharding_list(
        ctx, [op(s, dim=1, keepdim=True).values for s in t1_f32_sharding], num_partitions=3
    )


@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div, torch.rsub],
)
def test_binary_bc_dtensor(ctx, op):
    t1 = [torch.rand((2, 4, 5)) for _ in range(3)]
    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3)

    t2 = [torch.rand((2, 1, 5)) for _ in range(3)]
    dt2 = DTensor.from_sharding_list(ctx, t2, num_partitions=3)

    assert op(dt1, dt2) == DTensor.from_sharding_list(ctx, [op(s1, s2) for s1, s2 in zip(t1, t2)], num_partitions=3)

    t1 = [torch.rand((2, 4, 5)) for _ in range(3)]
    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3, axis=1)

    t2 = [torch.rand((4, 5)) for _ in range(3)]
    dt2 = DTensor.from_sharding_list(ctx, t2, num_partitions=3, axis=0)

    assert op(dt1, dt2) == DTensor.from_sharding_list(ctx, [op(s1, s2) for s1, s2 in zip(t1, t2)], num_partitions=3)


@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div, torch.rsub],
)
def test_binary_bc_tensor(ctx, op):
    t1 = [torch.rand((2, 3, 4, 5)) for _ in range(3)]
    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3)

    t2 = torch.rand((4, 5))
    assert op(dt1, t2) == DTensor.from_sharding_list(ctx, [op(s, t2) for s in t1], num_partitions=3)

    t2 = torch.rand((1, 1, 4, 5))
    assert op(dt1, t2) == DTensor.from_sharding_list(ctx, [op(s, t2) for s in t1], num_partitions=3)

    t1 = [torch.rand((2, 3, 4, 5)) for _ in range(3)]
    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3, axis=1)

    t2 = torch.rand((4, 5))
    assert op(dt1, t2) == DTensor.from_sharding_list(ctx, [op(s, t2) for s in t1], num_partitions=3)


def test_slice(ctx):
    t1 = [torch.rand((2, 3, 4, 5)) for _ in range(3)]
    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3)
    assert torch.allclose(torch.slice_f(dt1, 3), t1[1][1])

    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3, axis=1)
    assert torch.slice_f(dt1, 1) == DTensor.from_sharding_list(ctx, [s[1] for s in t1], num_partitions=3)

    dt1 = DTensor.from_sharding_list(ctx, t1, num_partitions=3)
    # assert torch.allclose(torch.slice_f(dt1, [3,1,2]), torch.cat(t1)[[3,1,2]])
    print(torch.slice_f(dt1, [3, 1, 2]).shape)
    print(torch.cat(t1)[[3, 1, 2]].shape)
