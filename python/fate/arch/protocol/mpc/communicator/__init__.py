from .communicator import Communicator


def get():
    if not Communicator.is_initialized():
        raise RuntimeError("Crypten not initialized. Please call crypten.init() first.")

    return Communicator.get()


def _init(ctx, init_ttp=False):
    global __tls
    from .communicator import Communicator

    if Communicator.is_initialized():
        return

    Communicator.initialize(ctx, init_ttp=init_ttp)


def uninit():
    Communicator.shutdown()


def is_initialized():
    return Communicator.is_initialized()
