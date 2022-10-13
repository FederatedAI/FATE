from fate.arch import tensor
import torch
from fate.arch import tensor
import torch
from fate.arch import Backend, Context

ctx = Context(
    "guest",
    backend=Backend.STANDALONE,
)
t1 = tensor.tensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
t2 = tensor.tensor(torch.tensor([[7, 8, 9], [10, 11, 12]]))
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

encryptor, decryptor = ctx.cipher.phe.keygen()
t3 = encryptor.encrypt(t1)
t4 = t3 + t2
t5 = decryptor.decrypt(t4)
print(t1)
print(t2)
print(t5)
