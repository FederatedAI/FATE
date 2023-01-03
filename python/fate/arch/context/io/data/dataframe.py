from ....unify import URI
from .df import Dataframe


class DataFrameWriter:
    def __init__(self, ctx, name: str, uri, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def write_dataframe(self, df):
        import json

        from fate.arch import dataframe

        with open(self.uri, "w") as f:
            data_dict = dataframe.serialize(self.ctx, df)
            json.dump(data_dict, f)


class DataFrameReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_dataframe(self):
        import json

        from fate.arch import dataframe

        with open(self.uri.path, "r") as fin:
            data_dict = json.loads(fin.read())
            df = dataframe.deserialize(self.ctx, data_dict)

        return Dataframe(df, df.shape[1], df.shape[0])
