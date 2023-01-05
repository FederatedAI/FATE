from typing import Protocol

from fate.components import Artifact, DatasetArtifact, MetricArtifact, ModelArtifact

from ...unify import URI, EggrollURI


class Reader(Protocol):
    ...


class Writer(Protocol):
    ...


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

    def reader(self, ctx, artifact, **kwargs):
        name = artifact.name
        metadata = artifact.metadata
        if "metadata" in kwargs:
            metadata = kwargs["metadata"]
        for k, v in kwargs.items():
            if k not in ["name", "metadata"]:
                metadata[k] = v
        writer_format = metadata.get("format")
        if "name" in kwargs:
            name = kwargs["name"]

        if isinstance(artifact, MetricArtifact):
            uri = URI.from_string(artifact.uri)
            if uri.schema == "file":
                from .metric.file import FileMetricsReader

                return FileMetricsReader(ctx, name, uri, metadata)
            if uri.schema in ["http", "https"]:
                from .metric.http import HTTPMetricsReader

                return HTTPMetricsReader(ctx, name, uri, metadata)

        if isinstance(artifact, ModelArtifact):
            uri = URI.from_string(artifact.uri)
            if uri.schema == "file":
                from .model.file import FileModelReader

                return FileModelReader(ctx, name, uri, metadata)

            if uri.schema in ["http", "https"]:
                from .model.http import HTTPModelReader

                return HTTPModelReader(ctx, name, uri, metadata)

        if isinstance(artifact, DatasetArtifact):
            uri = URI.from_string(artifact.uri)
            if uri.schema == "file":
                if writer_format == "csv":
                    from .data.csv import CSVReader

                    return CSVReader(ctx, name, uri, metadata)

                elif writer_format == "dataframe":
                    from .data.file import FileDataFrameReader

                    return FileDataFrameReader(ctx, name, uri.to_schema(), {})
            elif uri.schema == "eggroll":
                if writer_format == "dataframe":
                    from .data.eggroll import EggrollDataFrameReader

                    return EggrollDataFrameReader(ctx, uri.to_schema(), {})
        raise NotImplementedError(f"{artifact}")

    def writer(self, ctx, artifact: Artifact, **kwargs) -> "Writer":
        name = artifact.name
        metadata = artifact.metadata
        if "metadata" in kwargs:
            metadata = kwargs["metadata"]
        for k, v in kwargs.items():
            if k not in ["name", "metadata"]:
                metadata[k] = v
        writer_format = metadata.get("format")
        if "name" in kwargs:
            name = kwargs["name"]

        if isinstance(artifact, MetricArtifact):
            uri = URI.from_string(artifact.uri)
            if uri.schema == "file":
                from .metric.file import FileMetricsWriter

                return FileMetricsWriter(ctx, name, uri, metadata)

            if uri.schema in ["http", "https"]:
                from .metric.http import HTTPMetricsWriter

                return HTTPMetricsWriter(ctx, name, uri, metadata)
        if isinstance(artifact, ModelArtifact):
            uri = URI.from_string(artifact.uri)
            if uri.schema == "file":
                from .model.file import FileModelWriter

                return FileModelWriter(ctx, name, uri)
            if uri.schema in ["http", "https"]:
                from .model.http import HTTPModelWriter

                return HTTPModelWriter(ctx, name, uri, metadata)

        if isinstance(artifact, DatasetArtifact):
            uri = URI.from_string(artifact.uri)
            if uri.schema == "file":
                if writer_format == "csv":
                    from .data.csv import CSVWriter

                    return CSVWriter(ctx, name, uri, metadata)

                elif writer_format == "dataframe":
                    from .data.file import FileDataFrameWriter

                    return FileDataFrameWriter(ctx, name, uri.to_schema(), {})
            elif uri.schema == "eggroll":
                if writer_format == "dataframe":
                    from .data.eggroll import EggrollDataFrameWriter

                    return EggrollDataFrameWriter(ctx, uri.to_schema(), {})
        raise NotImplementedError(f"{artifact}")
