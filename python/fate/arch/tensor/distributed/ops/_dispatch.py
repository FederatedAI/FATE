_OPS = {}


def _register(func):
    _OPS[func.__name__] = func
    return func


def get_op(name):
    return _OPS.get(name)
