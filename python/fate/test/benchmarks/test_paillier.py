import torch
from fate.arch.protocol.phe.paillier import evaluator, keygen

sk, pk, coder = keygen(2048)
data = torch.rand(1000)
a = pk.encrypt_encoded(coder.encode_tensor(data), True)
b = pk.encrypt_encoded(coder.encode_tensor(data), True)


def test_iadd(benchmark):
    benchmark(lambda: evaluator.i_add(pk, a, b))
