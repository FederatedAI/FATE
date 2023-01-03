from ....unify import URI
from .df import Dataframe


class CSVReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_dataframe(self):
        import inspect

        from fate.arch import dataframe

        kwargs = {}
        p = inspect.signature(dataframe.CSVReader.__init__).parameters
        parameter_keys = p.keys()
        for k, v in self.metadata.items():
            if k in parameter_keys:
                kwargs[k] = v

        dataframe_reader = dataframe.CSVReader(**kwargs).to_frame(self.ctx, self.uri.path)
        # s_df = dataframe.serialize(self.ctx, dataframe_reader)
        # dataframe_reader = dataframe.deserialize(self.ctx, s_df)
        return Dataframe(dataframe_reader, dataframe_reader.shape[1], dataframe_reader.shape[0])


class CSVWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def write_dataframe(self, df):
        ...
