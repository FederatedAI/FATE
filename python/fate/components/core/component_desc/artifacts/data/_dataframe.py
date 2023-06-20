import logging
import typing

from fate.components.core.essential import DataframeArtifactType

from .._base_type import ArtifactDescribe, Metadata, _ArtifactType, _ArtifactTypeWriter

if typing.TYPE_CHECKING:
    from fate.arch import URI, Context
    from fate.arch.dataframe import DataFrame

logger = logging.getLogger(__name__)


class DataframeWriter(_ArtifactTypeWriter):
    def write(self, ctx, df: "DataFrame", name=None, namespace=None):
        logger.debug(f"start writing dataframe to artifact: {self.artifact}, name={name}, namespace={namespace}")
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
        # save data overview
        count = df.count()
        samples = df.data_overview()
        from fate.components.core.spec.artifact import DataOverview

        self.artifact.metadata.data_overview = DataOverview(count=count, samples=samples)

        logger.debug(f"write dataframe to artifact: {self.artifact}")


class DataframeArtifactDescribe(ArtifactDescribe):
    def get_type(self):
        return DataframeArtifactType

    def get_writer(self, uri: "URI", metadata: Metadata) -> _ArtifactTypeWriter:
        return DataframeWriter(_ArtifactType(uri, metadata))

    def _load_as_component_execute_arg(self, ctx: "Context", artifact: _ArtifactType):
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
