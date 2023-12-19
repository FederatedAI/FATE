import os

from ._serdes_base import p_dumps, p_loads


def get_unrestricted_serdes():
    if True or os.environ.get("SERDES_DEBUG_MODE") == "1":
        return UnrestrictedSerdes
    else:
        raise PermissionError("UnsafeSerdes is not allowed in production mode")


class UnrestrictedSerdes:
    @staticmethod
    def serialize(obj) -> bytes:
        return p_dumps(obj)

    @staticmethod
    def deserialize(bytes) -> object:
        return p_loads(bytes)
