from typing import Protocol, overload

from ..unify import URI


class Reader:
    @overload
    def __init__(self, ctx, uri: str, **kwargs):
        ...

    @overload
    def __init__(self, ctx, data, **kwargs):
        ...

    def __init__(self, ctx, *args, **kwargs):
        self.ctx = ctx
        if isinstance(args[0], str):
            self.uri = args[0]
            self.name = kwargs.get("name", "")
            self.metadata = kwargs.get("metadata", {})
        elif hasattr(args[0], "uri"):
            self.uri = args[0].uri
            self.name = args[0].name
            self.metadata = args[0].metadata
        else:
            raise ValueError(f"invalid arguments: {args} and {kwargs}")

    def read_dataframe(self):
        from fate.arch import dataframe

        self.data = dataframe.CSVReader(
            id_name="id", label_name="y", label_type="float32", delimiter=",", dtype="float32"
        ).to_frame(self.ctx, self.uri)
        return self


class IOKit:
    def reader(self, ctx, arg, **kwargs):
        name = ""
        metadata = {}
        if hasattr(arg, "uri"):
            uri = arg.uri
            name = arg.name
            metadata = arg.metadata
        elif isinstance(arg[0], str):
            uri = arg[0]
        else:
            raise ValueError(f"invalid arguments: {arg} and {kwargs}")
        if "name" in kwargs:
            name = kwargs["name"]
        if "metadata" in kwargs:
            metadata = kwargs["metadata"]
        for k, v in kwargs.items():
            if k not in ["name", "metadata"]:
                metadata[k] = v

        self.uri = URI.from_string(uri)
        format = metadata.get("format")
        if format is None:
            raise ValueError(f"reader format `{format}` unknown")
        return get_reader(format, ctx, name, uri, metadata)


class Reader(Protocol):
    ...


def get_reader(format, ctx, name, uri, metadata) -> Reader:
    if format == "csv":
        return CSVReader(ctx, name, uri, metadata)


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

        dataframe_reader = dataframe.CSVReader(**kwargs).to_frame(self.ctx, self.uri)
        s_df = dataframe.serialize(self.ctx, dataframe_reader)
        dataframe_reader = dataframe.deserialize(self.ctx, s_df)
        return DataframeReader(dataframe_reader, self.metadata["num_features"], self.metadata["num_samples"])


class DataframeReader:
    def __init__(self, frames, num_features, num_samples) -> None:
        self.data = frames
        self.num_features = num_features
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def to_local(self):
        return self.data.to_local()


class LibSVMReader:
    def read(self):
        ...

    def read_dataframe(self):
        ...


class CSVWriter:
    ...


class LibSVMWriter:
    ...
