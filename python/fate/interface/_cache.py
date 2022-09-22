from typing import List, Protocol


class Cache(Protocol):
    cache: List

    def add_cache(self, key, value):
        ...
