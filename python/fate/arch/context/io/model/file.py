import json

from ....unify import URI


class FileModelWriter:
    def __init__(self, ctx, name: str, uri: URI) -> None:
        self.ctx = ctx
        self.name = name
        self.path = uri.path

    def write_model(self, model):
        with open(self.path, "w") as f:
            json.dump(model, f)


class FileModelReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_model(self):
        with open(self.uri.path, "r") as fin:
            return json.loads(fin.read())
