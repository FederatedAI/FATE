import torch
from fate.arch.computing.backends.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.backends.standalone import StandaloneFederation
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
def t3():
    return torch.tensor([[1.0], [1.0], [1.0]])


@fixture
def t2(ctx):
    return DTensor.from_sharding_list(
        ctx,
        [
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ],
        axis=1,
    )


@fixture
def t4():
    return torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )


def test_local():
    # (2 x 3) @ (3 x 2) -> (2 x 2)
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    assert torch.allclose(torch.matmul(a, b), torch.rmatmul_f(b, a))


def test_shape_1_1(ctx):
    a = [torch.rand(9), torch.rand(10), torch.rand(11)]
    b = [torch.rand(9), torch.rand(10), torch.rand(11)]
    da = make_dist(ctx, a)
    db = make_dist(ctx, b)
    la = torch.concat(a)
    lb = torch.concat(b)
    # distributed
    assert torch.allclose(torch.matmul(da, db), torch.matmul(la, lb))
    assert torch.allclose(torch.rmatmul_f(da, db), torch.rmatmul_f(la, lb))
    # local
    assert torch.allclose(torch.matmul(da, lb), torch.matmul(la, lb))
    assert torch.allclose(torch.rmatmul_f(da, lb), torch.rmatmul_f(la, lb))


def test_shape_1_x(ctx):
    a = [torch.rand(9), torch.rand(10), torch.rand(11)]
    b = [torch.rand(9), torch.rand(10), torch.rand(11)]
    c = torch.rand(30, 4)
    d = torch.rand(4, 30)
    e = [torch.rand(9, 5), torch.rand(10, 5), torch.rand(11, 5)]
    f = [torch.rand(5, 9), torch.rand(5, 10), torch.rand(5, 11)]
    da = make_dist(ctx, a)
    db = make_dist(ctx, b)
    la = torch.concat(a)
    lb = torch.concat(b)
    de = make_dist(ctx, e)
    df = make_dist(ctx, f, axis=1)
    le = torch.concat(e)
    lf = torch.concat(f, dim=1)
    # distributed
    assert torch.allclose(torch.matmul(da, de), torch.matmul(la, le))
    assert torch.allclose(torch.rmatmul_f(da, df), torch.rmatmul_f(la, lf))

    # local
    assert torch.allclose(torch.matmul(da, c), torch.matmul(la, c))
    assert torch.allclose(torch.rmatmul_f(da, d), torch.rmatmul_f(la, d))


def test_shape_x_1(ctx):
    a_30 = [torch.rand(9), torch.rand(10), torch.rand(11)]
    b_30 = [torch.rand(9), torch.rand(10), torch.rand(11)]
    c = torch.rand(30, 4)
    d = torch.rand(4, 30)
    e_30_5 = [torch.rand(9, 5), torch.rand(10, 5), torch.rand(11, 5)]
    f_5_30 = [torch.rand(5, 9), torch.rand(5, 10), torch.rand(5, 11)]
    da_30 = make_dist(ctx, a_30)
    db_30 = make_dist(ctx, b_30)
    la_30 = torch.concat(a_30)
    lb_30 = torch.concat(b_30)
    de_30_5 = make_dist(ctx, e_30_5)
    df_5_30 = make_dist(ctx, f_5_30, axis=1)
    le_30_5 = torch.concat(e_30_5)
    lf_5_30 = torch.concat(f_5_30, dim=1)
    # distributed
    assert torch.allclose(torch.matmul(df_5_30, da_30), torch.matmul(lf_5_30, la_30))
    assert torch.allclose(torch.rmatmul_f(de_30_5, db_30), torch.rmatmul_f(le_30_5, lb_30))

    # local
    assert torch.allclose(torch.matmul(df_5_30, la_30), torch.matmul(lf_5_30, la_30))
    assert torch.allclose(torch.rmatmul_f(de_30_5, lb_30), torch.rmatmul_f(le_30_5, lb_30))


def test_shape_x_x_dist_dist_bc_matmul(ctx):
    e_30_5_13 = [torch.rand(9, 5, 13), torch.rand(10, 5, 13), torch.rand(11, 5, 13)]
    e_30_13_15 = [torch.rand(9, 13, 15), torch.rand(10, 13, 15), torch.rand(11, 13, 15)]

    assert torch.matmul(make_dist(ctx, e_30_5_13), make_dist(ctx, e_30_13_15)) == make_dist(
        ctx, [torch.matmul(s1, s2) for s1, s2 in zip(e_30_5_13, e_30_13_15)]
    )

    assert torch.rmatmul_f(make_dist(ctx, e_30_13_15), make_dist(ctx, e_30_5_13)) == make_dist(
        ctx, [torch.rmatmul_f(s1, s2) for s1, s2 in zip(e_30_13_15, e_30_5_13)]
    )


def test_shape_x_x_dist_dist_matmul(ctx):
    e_5_13_30 = [torch.rand(5, 13, 9), torch.rand(5, 13, 10), torch.rand(5, 13, 11)]
    e_19_30_17 = [torch.rand(5, 9, 17), torch.rand(5, 10, 17), torch.rand(5, 11, 17)]

    assert torch.allclose(
        torch.matmul(make_dist(ctx, e_5_13_30, axis=2), make_dist(ctx, e_19_30_17, axis=1)),
        torch.matmul(torch.concat(e_5_13_30, 2), torch.concat(e_19_30_17, 1)),
    )

    assert torch.allclose(
        torch.rmatmul_f(make_dist(ctx, e_19_30_17, axis=1), make_dist(ctx, e_5_13_30, axis=2)),
        torch.rmatmul_f(torch.concat(e_19_30_17, 1), torch.concat(e_5_13_30, 2)),
    )


def test_shape_x_x_dist_local_matmul(ctx):
    e_5_30_13 = [torch.rand(5, 9, 13), torch.rand(5, 10, 13), torch.rand(5, 11, 13)]
    e_5_13_30 = [torch.rand(5, 13, 9), torch.rand(5, 13, 10), torch.rand(5, 13, 11)]
    el_5_13_30 = torch.concat(e_5_13_30, dim=2)
    el_5_30_13 = torch.concat(e_5_30_13, dim=1)

    assert torch.matmul(make_dist(ctx, e_5_30_13, axis=1), el_5_13_30) == make_dist(
        ctx, [torch.matmul(s1, el_5_13_30) for s1 in e_5_30_13], axis=1
    )
    assert torch.allclose(
        torch.matmul(make_dist(ctx, e_5_13_30, axis=2), el_5_30_13), torch.matmul(el_5_13_30, el_5_30_13)
    )
    assert torch.rmatmul_f(make_dist(ctx, e_5_13_30, axis=2), el_5_30_13) == make_dist(
        ctx, [torch.rmatmul_f(s1, el_5_30_13) for s1 in e_5_13_30], axis=2
    )
    assert torch.allclose(
        torch.rmatmul_f(make_dist(ctx, e_5_30_13, axis=1), el_5_13_30), torch.rmatmul_f(el_5_30_13, el_5_13_30)
    )


def make_dist(ctx, tensors, axis=0):
    return DTensor.from_sharding_list(ctx, tensors, axis=axis)
