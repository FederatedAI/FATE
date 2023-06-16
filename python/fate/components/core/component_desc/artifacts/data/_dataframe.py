import typing

from .._base_type import URI, ArtifactDescribe, ArtifactType, Metadata

if typing.TYPE_CHECKING:
    from fate.arch.dataframe._dataframe import DataFrame


class DataframeArtifactType(ArtifactType):
    type = "dataframe"

    class EggrollAddress:
        def __init__(self, name: str, namespace: str, metadata: dict):
            self.name = name
            self.namespace = namespace
            self.metadata = metadata

        def to_uri_str(self):
            return f"eggroll://{self.namespace}/{self.name}"

        def read(self, ctx):
            from fate.arch import dataframe
            from fate.arch.computing._address import EggRollAddress

            table = ctx.computing.load(
                address=EggRollAddress(name=self.name, namespace=self.namespace),
            )
            table.schema = self.metadata
            return dataframe.deserialize(ctx, table)

        def write(self, ctx, df):
            from fate.arch import dataframe
            from fate.arch.computing._address import EggRollAddress

            table = dataframe.serialize(ctx, df)
            table.save(
                address=EggRollAddress(name=self.name, namespace=self.namespace),
                schema=self.metadata,
                partitions=table.partitions,
            )

    class HdfsAddress:
        def __init__(self, path: str, metadata: dict):
            self.path = path
            self.metadata = metadata

        def to_uri_str(self):
            return f"hdfs://{self.path}"

        def read(self, ctx):
            raise NotImplementedError()

    class FileAddress:
        def __init__(self, path: str, metadata: dict):
            self.path = path
            self.metadata = metadata

        def to_uri_str(self):
            return f"file://{self.path}"

        def read(self, ctx):
            from fate.arch.context.io.data.csv import CSVReader

            return CSVReader(ctx, self.path, self.metadata).read_dataframe()

        def write(self, ctx, dataframe):
            # from fate.arch.context.io.data.csv import CSVWriter
            #
            # return CSVWriter(ctx, self.path, self.metadata).write_dataframe(dataframe)
            from fate.arch.context.io.data.file import FileDataFrameWriter

            writer = FileDataFrameWriter(ctx, self.path, self.metadata)
            writer.write_dataframe(dataframe)
            self.metadata = writer.metadata

    def __init__(self, metadata: Metadata, address):
        self.metadata = metadata
        self.address = address

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        schema = uri.schema
        if schema == "file":
            address = cls.FileAddress(uri.path, metadata.metadata)
        elif schema == "hdfs":
            address = cls.HdfsAddress(uri.path, metadata.metadata)
        elif schema == "eggroll":
            _, namespace, name = uri.path.split("/")
            address = cls.EggrollAddress(name, namespace, metadata.metadata)
        else:
            raise ValueError(f"unsupported schema {schema}")
        return cls(metadata, address)

    def dict(self):
        return {
            "metadata": self.metadata,
            "uri": self.address.to_uri_str(),
        }


class DataframeWriter:
    def __init__(self, artifact: DataframeArtifactType) -> None:
        self.artifact = artifact

    def write(self, ctx, dataframe: "DataFrame", name=None, namespace=None):
        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace
        self.artifact.address.write(ctx, dataframe)

    def __str__(self):
        return f"DataframeWriter({self.artifact})"

    def __repr__(self):
        return str(self)


class DataframeArtifactDescribe(ArtifactDescribe[DataframeArtifactType]):
    def _get_type(self):
        return DataframeArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: DataframeArtifactType):
        return artifact.address.read(ctx)

    def _load_as_component_execute_arg_writer(self, ctx, artifact: DataframeArtifactType):
        return DataframeWriter(artifact)
