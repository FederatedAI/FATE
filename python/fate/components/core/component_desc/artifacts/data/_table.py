from typing import Union

from fate.components.core.essential import TableArtifactType

from .._base_type import URI, ArtifactDescribe, Metadata, _ArtifactType


class _TableArtifactType(_ArtifactType["TableWriter"]):
    type = TableArtifactType

    class EggrollAddress:
        def __init__(self, name: str, namespace: str, metadata: dict):
            self.name = name
            self.namespace = namespace
            self.metadata = metadata

        def to_uri_str(self):
            return f"eggroll://{self.namespace}/{self.name}"

        def read(self, ctx):
            from fate.arch.computing import EggRollAddress

            table = ctx.computing.load(
                address=EggRollAddress(name=self.name, namespace=self.namespace),
                schema=self.metadata,
            )
            table.schema = self.metadata
            return table

        def write(self, ctx, table):
            from fate.arch.computing import EggRollAddress

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
            from fate.arch.computing import HDFSAddress

            table = ctx.computing.load(
                address=HDFSAddress(path=self.path),
            )
            table.schema = self.metadata
            return table

        def write(self, ctx, table):
            from fate.arch.computing import HDFSAddress

            table.save(
                address=HDFSAddress(path=self.path),
                schema=self.metadata,
                partitions=table.partitions,
            )

    def __init__(self, metadata: Metadata, address: Union[EggrollAddress, HdfsAddress]):
        self.metadata = metadata
        self.address = address

    @classmethod
    def _load(cls, uri: URI, metadata: Metadata):
        schema = uri.schema
        if schema == "hdfs":
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

    def get_writer(self) -> "TableWriter":
        return TableWriter(self)


class TableWriter:
    def __init__(self, artifact: _TableArtifactType) -> None:
        self.artifact = artifact

    def write(self, slot):
        ...

    def __str__(self):
        return f"TableWriter({self.artifact})"

    def __repr__(self):
        return self.__str__()


class TableArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return _TableArtifactType

    def _load_as_component_execute_arg(self, ctx, artifact: _TableArtifactType):
        return artifact.address.read(ctx)
