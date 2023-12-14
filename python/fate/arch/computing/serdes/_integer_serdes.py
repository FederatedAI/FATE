def get_integer_serdes():
    return IntegerSerdes()


class IntegerSerdes:
    def __init__(self):
        ...

    def serialize(self, obj) -> bytes:
        return obj.to_bytes(8, "big")

    def deserialize(self, bytes) -> object:
        return int.from_bytes(bytes, "big")
