from typing import Callable, Protocol, Union

LOGMSG = Union[str, Callable[[], str]]


class Logger(Protocol):
    def info(self, msg: LOGMSG):
        ...

    def debug(self, msg: LOGMSG):
        ...

    def error(self, msg: LOGMSG):
        ...

    def warning(self, msg: LOGMSG):
        ...
