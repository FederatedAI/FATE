class OpsDispatchException(Exception):
    ...


class OpsDispatchBadSignatureError(OpsDispatchException):
    ...


class OpsDispatchBadDtypeError(OpsDispatchException):
    ...


class OpsDispatchUnsupportedError(OpsDispatchException):
    def __init__(self, method, distributed, device, dtype) -> None:
        super().__init__(f"method={method}, distributed={distributed}, device={device}, dtype={dtype}")


class OpDispatchInvalidDevice(OpsDispatchException):
    ...
