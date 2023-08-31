# computing = CSession()
#
# arbiter = ("arbiter", "10000")
# guest = ("guest", "10000")
# host = ("host", "9999")
# name = "fed"
# ctx = Context(computing=computing, federation=StandaloneFederation(computing, name, guest, [guest, host, arbiter]))
# kit = ctx.cipher.phe.setup(options={"kind": "paillier", "key_length": 1024})
# sk, pk, coder, evaluator, encryptor = kit.sk, kit.pk, kit.coder, kit.evaluator, kit.get_tensor_encryptor()
import cProfile

import torch
from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.histogram import Histogram

computing = CSession()
ctx = Context(computing=computing)
kit = ctx.cipher.phe.setup(options={"kind": "paillier", "key_length": 2048})


# def test_baseline():

hist = Histogram.create(
    1,
    [3, 2],
    {"c0": {"type": "paillier", "pk": kit.pk, "evaluator": kit.evaluator, "coder": kit.coder, "stride": 1000}},
)
fids = torch.tensor([[1, 0], [0, 1], [2, 1], [2, 0]])
nids = torch.tensor([[0], [0], [0], [0]])
values = {
    "c0": kit.get_tensor_encryptor().encrypt_encoded(
        kit.get_tensor_coder().encode(torch.rand(4, 1000)),
        False,
    )
}


hist.i_update(fids, nids, values)


def my_test():
    import time

    start = time.time()
    for _ in range(1):
        hist.i_update(fids, nids, values)
    print(time.time() - start)


profiler = cProfile.Profile()
profiler.runcall(my_test)
profiler.dump_stats("my_profile.prof")
# print(hist.decrypt({"c0": kit.sk}).decode({"c0": (kit.coder, torch.float32)}))
