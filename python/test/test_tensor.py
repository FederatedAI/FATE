import torch
from fate.arch import Backend, Context, tensor
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context, disable_inner_logs
from fate.arch.federation.standalone import StandaloneFederation

# disable_inner_logs()
computing = CSession()
ctx = Context(
    "guest",
    backend=Backend.STANDALONE,
    computing=computing,
    federation=StandaloneFederation(
        computing, "fed", ("guest", 10000), [("host", 9999)]
    ),
)
computing2 = CSession()
ctx2 = Context(
    "guest",
    backend=Backend.STANDALONE,
    computing=computing2,
    federation=StandaloneFederation(
        computing2, "fed", ("host", 9999), [("guest", 10000)]
    ),
)
t1 = tensor.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
t2 = tensor.tensor(torch.tensor([[7, 8, 9], [10, 11, 12]]))
ctx.hosts.push("t1", t1)
t3 = ctx2.guest.pull("t1").unwrap()
print(t3)
# print(f"t1={t1}")
# print(f"t2={t2}")
# for method in ["add", "sub", "div", "mul"]:
#     out = getattr(tensor, method)(t1, t2)
#     print(f"{method}(t1, t2): \n{out}")

# for method in ["exp", "log", "neg", "reciprocal", "square", "abs"]:
#     out = getattr(tensor, method)(t1)
#     print(f"{method}(t1): \n{out}")

# for method in ["pow", "remainder"]:
#     out = getattr(tensor, method)(t1, 3.14)
#     print(f"{method}(t1, 3.14): \n{out}")

# encryptor, decryptor = ctx.cipher.phe.keygen()
# t3 = encryptor.encrypt(t1)

# t4 = t3 + t2
# t5 = decryptor.decrypt(t4)
# print(t1)
# print(t2)
# print(t5)
