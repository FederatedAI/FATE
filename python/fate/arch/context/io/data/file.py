from ....unify import FileURI
from .df import Dataframe


class FileDataFrameWriter:
    def __init__(self, ctx, name: str, uri: FileURI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = FileMetaURI(uri)
        self.metadata = metadata

    def write_dataframe(self, df):
        import json

        from fate.arch import dataframe

        table = dataframe.serialize(self.ctx, df)
        with open(self.uri.get_data_path(), "w") as f:
            json.dump(list(table.collect()), f)
        with open(self.uri.get_meta_path(), "w") as f:
            json.dump(table.schema, f)


class FileDataFrameReader:
    def __init__(self, ctx, name: str, uri: FileURI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = FileMetaURI(uri)
        self.metadata = metadata

    def read_dataframe(self):
        import json

        from fate.arch import dataframe

        with open(self.uri.get_meta_path(), "r") as fin:
            schema = json.load(fin)
        with open(self.uri.get_data_path(), "r") as fin:
            data = json.load(fin)

        table = self.ctx.computing.parallelize(data, include_key=True, partition=1)
        data.schema = schema
        df = dataframe.deserialize(self.ctx, table)

        return Dataframe(df, df.shape[1], df.shape[0])


class FileMetaURI:
    def __init__(self, uri: FileURI) -> None:
        self.uri = uri

    def get_data_path(self):
        return self.uri.path

    def get_meta_path(self):
        return f"{self.uri.path}.meta"
