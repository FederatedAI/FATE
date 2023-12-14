def mmh3_partitioner(key: bytes, total_partitions):
    import mmh3

    return mmh3.hash(key) % total_partitions

