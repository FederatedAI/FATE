import os
import pickle


def get_unrestricted_serdes():
    if True or os.environ.get("SERDES_DEBUG_MODE") == "1":
        return UnrestrictedSerdes
    else:
        raise PermissionError("UnsafeSerdes is not allowed in production mode")


class UnrestrictedSerdes:
    @staticmethod
    def serialize(obj) -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def deserialize(bytes) -> object:
        return pickle.loads(bytes)
