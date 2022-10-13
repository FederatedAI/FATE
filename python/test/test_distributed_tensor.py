if __name__ == "__main__":

    from fate.arch import tensor
    import torch
    from fate.arch import Backend, Context
    from fate.arch.computing.standalone import CSession
    from fate.arch.context import Context, disable_inner_logs
    import pprint

    disable_inner_logs()
    computing = CSession()
    ctx = Context(
        "guest",
        backend=Backend.STANDALONE,
        computing=computing,
    )
    t1 = tensor.distributed_tensor(
        ctx,
        [
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ],
    )
    # t2 = tensor.distributed_tensor(
    #     ctx,
    #     [
    #         torch.tensor([[1, 2, 3], [4, 5, 6]]),
    #         torch.tensor([[1, 2, 3], [4, 5, 6]]),
    #         torch.tensor([[1, 2, 3], [4, 5, 6]]),
    #     ],
    # )
    # pprint.pprint(f"t1={t1.to_local()}")
    # pprint.pprint(f"t2={t2.to_local()}")

    # for method in ["add", "sub", "div", "mul"]:
    #     out = getattr(tensor, method)(t1, t2)
    #     pprint.pprint(f"{method}(t1, t2):\n{out.to_local()}")

    # for method in ["exp", "log", "neg", "reciprocal", "square", "abs"]:
    #     out = getattr(tensor, method)(t1)
    #     pprint.pprint(f"{method}(t1):\n{out.to_local()}")

    encryptor, decryptor = ctx.cipher.phe.keygen()
    t2 = encryptor.encrypt(t1)
    t3 = t2 + t1
    t4 = t2 * t1
    t5 = decryptor.decrypt(t3)
    t6 = decryptor.decrypt(t4)
    print(t1.storage.collect())
    print(t5.storage.collect())
    print(t6.storage.collect())
