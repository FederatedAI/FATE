from .paillier import _RustPaillierStorage


def impl_op_for(storage, name):
    return isinstance(storage, _RustPaillierStorage) and has_op(name)


def has_op(name):
    return name in ops


def get_ops(name):
    return ops.get(name)


def encrypt(storage, decryptor):
    return decryptor.decrypt(storage)


ops = {"encrypt": encrypt}
