from .impl import cpu_paillier

# def dispatch_unary(name, storage, args, kwargs):
#     if impl_op_for(storage, name):
#         return get_ops(name)(storage, *args, **kwargs)
#     raise NotImplementedError(f"storage `{storage}` not implemente {name}")


# def dispatch_binary(name, storage_a, storage_b, args, kwargs):
#     from .impl.torch_based import _TorchStorage, _TorchStorageOps

#     if isinstance(storage_a, _TorchStorage) or isinstance(storage_b, _TorchStorage):
#         if op := _TorchStorageOps.get_op(name):
#             return op(storage_a, storage_b, *args, **kwargs)
#     raise NotImplementedError(f"storage `{storage_a}, {storage_b}` not implemente {name}")


# def dispatch_encrypt(storage, encryptor):
#     if impl_op_for(storage, "encrypt"):
#         return get_ops("encrypt")(storage, encryptor)
#     raise NotImplementedError(f"storage `{storage}` not implemente encrypt")


# def dispatch_decrypt(storage, decryptor):
#     if cpu_paillier.impl_op_for(storage, "decryptor"):
#         return cpu_paillier.get_ops("decrypt")(storage, decryptor)

#     raise NotImplementedError(f"storage `{storage}` not implemente encrypt")


def _get_impl(*args, **kwargs):
    # TODO: dispatch rule
    from .impl.cpu_paillier.paillier import _RustPaillierStorage

    for v in args:
        if isinstance(v, _RustPaillierStorage):
            from .impl.cpu_paillier import _ops as ops

            return ops
    for _, v in kwargs.items():
        if isinstance(v, _RustPaillierStorage):
            from .impl.cpu_paillier import _ops as ops

            return ops

    from fate.arch.storage.impl.torch_based import _ops as ops

    return ops


def _auto_dispatch(func):
    name = func.__name__

    def wraped(*args, **kwargs):
        ops = _get_impl(*args, **kwargs)
        op = getattr(ops, name)
        if op is None:
            raise NotImplementedError(f"op `{name}` not found in {ops}")
        return op(*args, **kwargs)

    return wraped


def _auto_dispatch_encrypt(func):
    def wraped(s, encryptor):
        return encryptor.encrypt(s)

    return wraped


def _auto_dispatch_decrypt(func):
    def wraped(s, decryptor):
        return decryptor.decrypt(s)

    return wraped
