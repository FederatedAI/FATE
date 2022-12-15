import logging
from typing import Protocol, overload

from ..unify import URI, HttpURI, HttpsURI


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
    @staticmethod
    def _parse_args(arg, **kwargs):
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

        uri = URI.from_string(uri)
        format = metadata.get("format")
        return format, name, uri, metadata

    def reader(self, ctx, arg, **kwargs) -> "Reader":
        format, name, uri, metadata = self._parse_args(arg, **kwargs)
        if format is None:
            raise ValueError(f"reader format `{format}` unknown")
        return get_reader(format, ctx, name, uri, metadata)

    def writer(self, ctx, arg, **kwargs) -> "Writer":
        format, name, uri, metadata = self._parse_args(arg, **kwargs)
        if format is None:
            raise ValueError(f"writer format `{format}` unknown")
        return get_writer(format, ctx, name, uri, metadata)


class Reader(Protocol):
    def read_dataframe(self):
        ...


def get_reader(format, ctx, name, uri, metadata) -> Reader:
    if format == "csv":
        return CSVReader(ctx, name, uri.path, metadata)

    if format == "dataframe":
        return DataFrameReader(ctx, name, uri.path, metadata)

    if format == "json":
        return JsonReader(ctx, name, uri, metadata)


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
        # s_df = dataframe.serialize(self.ctx, dataframe_reader)
        # dataframe_reader = dataframe.deserialize(self.ctx, s_df)
        return Dataframe(dataframe_reader, dataframe_reader.shape[1], dataframe_reader.shape[0])


class DataFrameReader:
    def __init__(self, ctx, name: str, uri: str, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_dataframe(self):
        import json

        from fate.arch import dataframe

        with open(self.uri, "r") as fin:
            data_dict = json.loads(fin.read())
            df = dataframe.deserialize(self.ctx, data_dict)

        return Dataframe(df, df.shape[1], df.shape[0])


class JsonReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_model(self):
        import json
        if isinstance(self.uri.to_schema(), HttpURI) or isinstance(self.uri.to_schema(), HttpsURI):
            import requests
            url = f"{self.uri.schema}://{self.uri.authority}{self.uri.path}"
            logging.debug(url)
            model_dict = requests.get(url=url).json().get("data", {})
            logging.debug(model_dict)
        elif isinstance(self.uri.to_schema(), URI):
            with open(self.uri.path, "r") as fin:
                model_dict = json.loads(fin.read())
        return model_dict


class LibSVMReader:
    def read(self):
        ...

    def read_dataframe(self):
        ...


def get_writer(format, ctx, name, uri, metadata) -> Reader:
    if format == "csv":
        return CSVWriter(ctx, name, uri, metadata)

    if format == "json":
        return JsonWriter(ctx, name, uri, metadata)

    if format == "dataframe":
        return DataFrameWriter(ctx, name, uri.path, metadata)


class Writer(Protocol):
    ...


class CSVWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def write_dataframe(self, df):
        ...


class JsonWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def write_model(self, model):
        import json
        if isinstance(self.uri.to_schema(), HttpURI) or isinstance(self.uri.to_schema(), HttpsURI):
            import requests
            url = f"{self.uri.schema}://{self.uri.authority}{self.uri.path}"
            logging.debug(url)
            response = requests.post(url=url, json={"data": model})
            logging.debug(response.text)
        elif isinstance(self.uri.to_schema(), URI):
            with open(self.uri.to_schema(), "w") as f:
                json.dump(model, f)

    def write_metric(self, metric):
        import json
        if isinstance(self.uri.to_schema(), URI):
            with open(self.uri.path, "w") as f:
                json.dump(metric, f)


class LibSVMWriter:
    ...


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


class Dataframe:
    def __init__(self, frames, num_features, num_samples) -> None:
        self.data = frames
        self.num_features = num_features
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def to_local(self):
        return self.data.to_local()
