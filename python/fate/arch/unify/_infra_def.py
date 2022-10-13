from enum import Enum


class device(Enum):
    def __init__(self, type: str, index) -> None:
        self.type = type
        self.index = index

    CPU = ("CPU", 1)
    CUDA = ("CUDA", 2)


class Backend(Enum):
    LOCAL = "LOCAL"
    STANDALONE = "STANDALONE"
    EGGROLL = "EGGROLL"
    SPARK = "SPARK"
