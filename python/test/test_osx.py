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

t1 = tensor.distributed_tensor(
    ctx,
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
ctx.hosts.put("tensor", t1)
t3 = ctx2.guest.get("tensor")
t3.storage.blocks.mapValues(lambda x: x)
print(t3.storage.collect())
