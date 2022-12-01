if __name__ == "__main__":

    import torch
    from fate.arch import Context, tensor
    from fate.arch.computing.standalone import CSession
    from fate.arch.context import Context
    from fate.arch.federation.standalone import StandaloneFederation

    # disable_inner_logs()
    def create_ctx(fed_id, local_party, parties):
        _computing = CSession()
        ctx = Context(
            "guest",
            computing=_computing,
            federation=StandaloneFederation(_computing, fed_id, local_party, parties),
        )
        return ctx

    ctx1 = create_ctx("fed", ("guest", 10), [("host", 9)])
    ctx2 = create_ctx("fed", ("host", 9), [("guest", 10)])

    t1 = tensor.distributed_tensor(
        ctx1,
        [
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ],
    )
    t2 = tensor.distributed_tensor(
        ctx2,
        [
            torch.tensor([[1.0, 2, 3], [4, 5, 6]]),
            torch.tensor([[1.0, 2, 3], [4, 5, 6]]),
            torch.tensor([[1.0, 2, 3], [4, 5, 6]]),
        ],
    )
    t3 = tensor.distributed_tensor(
        ctx2,
        [
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ],
        d_axis=1,
    )
    lt1 = tensor.tensor(torch.tensor([[1, 2, 3]]))
    print(t1[0])
    print(tensor.matmul(t3, t2))

    print(tensor.sum(t3))
    print(tensor.sum(t3, dim=1))

    print(tensor.mean(t3))
    print(tensor.mean(t3, dim=1))

    print(tensor.std(t3))
    print(tensor.std(t3, dim=1))

    print(tensor.var(t3))
    print(tensor.var(t3, dim=1))
