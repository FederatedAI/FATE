import hashlib


def partitioner(hash_func, total_partitions):
    def partition(key):
        return hash_func(key) % total_partitions

    return partition


def integer_partitioner(key: bytes, total_partitions):
    return int.from_bytes(key, "big") % total_partitions


def mmh3_partitioner(key: bytes, total_partitions):
    import mmh3

    return mmh3.hash(key) % total_partitions


def _java_string_like_partitioner(key, total_partitions):
    _key = hashlib.sha1(key).digest()
    _key = int.from_bytes(_key, byteorder="little", signed=False)
    b, j = -1, 0
    while j < total_partitions:
        b = int(j)
        _key = ((_key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = float(b + 1) * (float(1 << 31) / float((_key >> 33) + 1))
    return int(b)


def get_default_partitioner():
    return _java_string_like_partitioner


def get_partitioner_by_type(partitioner_type: int):
    if partitioner_type == 0:
        return get_default_partitioner()
    elif partitioner_type == 1:
        return integer_partitioner
    elif partitioner_type == 2:
        return mmh3_partitioner
    else:
        raise ValueError(f"partitioner type `{partitioner_type}` not supported")
