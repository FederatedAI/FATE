import torch
from fate.arch import Context, tensor
from fate.arch.computing.eggroll import CSession
from fate.arch.context import Context
from fate.arch.federation.eggroll import EggrollFederation

federation_id = "f-000001"
computing = CSession(session_id="guest")
ctx = Context(
    "guest",
    computing=computing,
    federation=EggrollFederation(
        rp_ctx=computing.get_rpc(),
        rs_session_id=federation_id,
        party=("guest", "10000"),
        parties=[("guest", "10000"), ("host", "9999")],
        proxy_endpoint="127.0.0.1:9370",
    ),
)
computing2 = CSession(session_id="host")
ctx2 = Context(
    "host",
    computing=computing2,
    federation=EggrollFederation(
        rp_ctx=computing2.get_rpc(),
        rs_session_id=federation_id,
        party=("host", "9999"),
        parties=[("guest", "10000"), ("host", "9999")],
        proxy_endpoint="127.0.0.1:9370",
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
