import hashlib


def _java_string_like_partitioner(key, total_partitions):
    _key = hashlib.sha1(key).digest()
    _key = int.from_bytes(_key, byteorder="little", signed=False)
    b, j = -1, 0
    while j < total_partitions:
        b = int(j)
        _key = ((_key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = float(b + 1) * (float(1 << 31) / float((_key >> 33) + 1))
    return int(b)
