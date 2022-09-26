from typing import Protocol


class GarbageCollector(Protocol):
    def register_clean_action(self, name: str, tag: str, obj, method: str, kwargs):
        ...

    def clean(self, name: str, tag: str):
        ...
