def integer_partitioner(key: bytes, total_partitions):
    return int.from_bytes(key, "big") % total_partitions
