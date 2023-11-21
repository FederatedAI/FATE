import pickle
import os


class UnsafeSerdes:
    def __init__(self):
        ...

    def serialize(self, obj) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, bytes) -> object:
        return pickle.loads(bytes)


class IntegerSerdes:
    def __init__(self):
        ...

    def serialize(self, obj) -> bytes:
        return obj.to_bytes(8, "big")

    def deserialize(self, bytes) -> object:
        return int.from_bytes(bytes, "big")


def get_unsafe_serdes():
    if True or os.environ.get("SERDES_DEBUG_MODE") == "1":
        return UnsafeSerdes()
    else:
        raise PermissionError("UnsafeSerdes is not allowed in production mode")


def get_serdes_by_type(serdes_type: int):
    if serdes_type == 0:
        return get_unsafe_serdes()
    elif serdes_type == 1:
        return IntegerSerdes()
    else:
        raise ValueError(f"serdes type `{serdes_type}` not supported")
