from typing import List, Protocol


class CpnOutput(Protocol):
    data: list
    model: dict
    cache: List[tuple]
