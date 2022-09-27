from enum import Enum


class Device(Enum):
    CPU = "CPU"
    GPU = "GPU"
    FPGA = "FPGA"


class Backend(Enum):
    LOCAL = "LOCAL"
    STANDALONE = "STANDALONE"
    EGGROLL = "EGGROLL"
    SPARK = "SPARK"
