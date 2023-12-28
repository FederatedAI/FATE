from .communicator import Communicator
import contextlib


def get():
    if not Communicator.is_initialized():
        raise RuntimeError("Crypten not initialized. Please call crypten.init() first.")

    return Communicator.get()
