from fate_arch.abc import CTableABC


def is_table(v):
    return isinstance(v, CTableABC)
