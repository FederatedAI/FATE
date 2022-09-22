from typing import Protocol


class Summary(Protocol):
    summary: dict

    def save(self):
        ...

    def reset(self, summary: dict):
        ...

    def add(self, key: str, value):
        ...
