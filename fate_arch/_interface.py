from typing_extensions import Protocol


class Rubbish(Protocol):
    """
    a collection collects all tables / objects in federation tagged by `tag`.
    """

    def add_table(self, table):
        ...

    def add_obj(self, table, key):
        ...
