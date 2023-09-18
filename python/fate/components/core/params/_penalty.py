from typing import Type

from ._fields import StringChoice


class Penalty(StringChoice):
    choice = {}


def penalty_param(l1=True, l2=True, none=True) -> Type[str]:
    choice = {"L1": l1, "L2": l2, "none": none}
    namespace = dict(
        choice={k for k, v in choice.items() if v},
    )
    return type("PenaltyValue", (Penalty,), namespace)
