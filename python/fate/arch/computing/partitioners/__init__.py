def partitioner(hash_func, total_partitions):
    def partition(key):
        return hash_func(key) % total_partitions

    return partition


def get_default_partitioner():
    from ._mmh3_partitioner import mmh3_partitioner

    return mmh3_partitioner


def get_partitioner_by_type(partitioner_type: int):
    if partitioner_type == 0:
        return get_default_partitioner()
    elif partitioner_type == 1:
        from ._integer_partitioner import integer_partitioner

        return integer_partitioner
    elif partitioner_type == 2:
        from ._mmh3_partitioner import mmh3_partitioner

        return mmh3_partitioner
    elif partitioner_type == 3:
        from ._java_string_like_partitioner import _java_string_like_partitioner

        return _java_string_like_partitioner
    else:
        raise ValueError(f"partitioner type `{partitioner_type}` not supported")
