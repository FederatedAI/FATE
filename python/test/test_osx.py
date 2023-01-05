import torch
from fate.arch import Context, tensor
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.osx import OSXFederation

federation_id = "f-000001"
computing = CSession()
ctx = Context(
    "guest",
    computing=computing,
    federation=OSXFederation.from_conf(
        federation_session_id=federation_id,
        computing_session=computing,
        party=("guest", "10000"),
        parties=[("guest", "10000"), ("host", "9999")],
        host="127.0.0.1",
        port=9370,
    ),
)
computing2 = CSession()
ctx2 = Context(
    "host",
    computing=computing2,
    federation=OSXFederation.from_conf(
        federation_session_id=federation_id,
        computing_session=computing,
        party=("host", "9999"),
        parties=[("guest", "10000"), ("host", "9999")],
        host="127.0.0.1",
        port=9370,
    ),
)
t1 = tensor.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
t2 = tensor.tensor(torch.tensor([[7, 8, 9], [10, 11, 12]]))
print(tensor.matmul(t1, t2.T))
ctx.hosts.put("ttt1", t1)
t3 = ctx2.guest.get("ttt1")
print(t3)
print(f"t1={t1}")
print(f"t2={t2}")
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
