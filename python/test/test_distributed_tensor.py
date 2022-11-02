if __name__ == "__main__":

    import torch
    from fate.arch import Backend, Context, tensor
    from fate.arch.computing.standalone import CSession
    from fate.arch.context import Context, disable_inner_logs
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
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ],
    )
    ctx1.hosts.put("tensor", t1)
    t3 = ctx2.guest.get("tensor")
    print(t3.storage.collect())

    # lt1 = tensor.tensor(torch.tensor([[1, 2, 3]]))
    # pprint.pprint(f"t1={t1.to_local()}")
    # pprint.pprint(f"t2={t2.to_local()}")
    # pprint.pprint(f"t1+lt1={(t1+lt1).to_local()}")

    # for method in ["add", "sub", "div", "mul"]:
    #     out = getattr(tensor, method)(t1, t2)
    #     pprint.pprint(f"{method}(t1, t2):\n{out.to_local()}")

    # for method in ["exp", "log", "neg", "reciprocal", "square", "abs"]:
    #     out = getattr(tensor, method)(t1)
    #     pprint.pprint(f"{method}(t1):\n{out.to_local()}")

    # encryptor, decryptor = ctx.cipher.phe.keygen()
    # t2 = encryptor.encrypt(t1)
    # t3 = t2 + t1
    # t4 = t2 * t1
    # t5 = decryptor.decrypt(t3)
    # t6 = decryptor.decrypt(t4)
    # print(t1.storage.collect())
    # print(t5.storage.collect())
    # print(t6.storage.collect())
