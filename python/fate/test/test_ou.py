from fate_utils.ou import Coder as _Coder

from fate.arch.protocol.phe.ou import *


def test_pack_float():
    offset_bit = 32
    precision = 16
    coder = Coder(_Coder())
    vec = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    packed = coder.pack_floats(vec, offset_bit, 2, precision)
    unpacked = coder.unpack_floats(packed, offset_bit, 2, precision, 5)
    assert torch.allclose(vec, unpacked, rtol=1e-3, atol=1e-3)


def test_pack_squeeze():
    offset_bit = 32
    precision = 16
    pack_num = 2
    pack_packed_num = 2
    vec1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    vec2 = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    sk, pk, coder = keygen(1024)
    a = coder.pack_floats(vec1, offset_bit, pack_num, precision)
    ea = pk.encrypt_encoded(a, obfuscate=False)
    b = coder.pack_floats(vec2, offset_bit, pack_num, precision)
    eb = pk.encrypt_encoded(b, obfuscate=False)
    ec = evaluator.add(ea, eb, pk)

    # pack packed encrypted
    ec_pack = evaluator.pack_squeeze(ec, pack_packed_num, offset_bit * 2, pk)
    c_pack = sk.decrypt_to_encoded(ec_pack)
    c = coder.unpack_floats(c_pack, offset_bit, pack_num * pack_packed_num, precision, 5)
    assert torch.allclose(vec1 + vec2, c, rtol=1e-3, atol=1e-3)
