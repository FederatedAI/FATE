from typing import Protocol


class Params(Protocol):
    is_need_run: bool
    is_need_cv: bool
    is_need_stepwise: bool
