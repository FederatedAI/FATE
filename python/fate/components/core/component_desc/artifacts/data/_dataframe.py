import typing

from fate.components.core.essential import DataframeArtifactType

from .._base_type import ArtifactDescribe, _ArtifactType

if typing.TYPE_CHECKING:
    from fate.arch import Context
    from fate.arch.dataframe import DataFrame


class DataframeWriter:
    def __init__(self, artifact: "_DataframeArtifactType") -> None:
        self.artifact = artifact

    def write(self, ctx, df: "DataFrame", name=None, namespace=None):
        from fate.arch import dataframe

        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace

        table = dataframe.serialize(ctx, df)
        if "schema" not in self.artifact.metadata.metadata:
            self.artifact.metadata.metadata["schema"] = {}
        table.save(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata["schema"],
            options=self.artifact.metadata.metadata.get("options", None),
        )

    def __str__(self):
        return f"DataframeWriter({self.artifact})"

    def __repr__(self):
        return self.__str__()


class _DataframeArtifactType(_ArtifactType[DataframeWriter]):
    type = DataframeArtifactType

    def get_writer(self) -> DataframeWriter:
        return DataframeWriter(self)


class DataframeArtifactDescribe(ArtifactDescribe[_DataframeArtifactType]):
    def get_type(self):
        return _DataframeArtifactType

    def _load_as_component_execute_arg(self, ctx: "Context", artifact: _DataframeArtifactType):
        from fate.arch import dataframe

        if artifact.uri.schema == "file":
            import inspect

            from fate.arch import dataframe

            kwargs = {}
            p = inspect.signature(dataframe.CSVReader.__init__).parameters
            parameter_keys = p.keys()
            for k, v in artifact.metadata.metadata.items():
                if k in parameter_keys:
                    kwargs[k] = v

            return dataframe.CSVReader(**kwargs).to_frame(ctx, artifact.uri.path)

        table = ctx.computing.load(
            uri=artifact.uri,
            schema=artifact.metadata.metadata.get("schema", None),
            options=artifact.metadata.metadata.get("options", None),
        )
        return dataframe.deserialize(ctx, table)
