from .communicator import Communicator


def get():
    if not Communicator.is_initialized():
        raise RuntimeError("Crypten not initialized. Please call crypten.init() first.")

    return Communicator.get()
