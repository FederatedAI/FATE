#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import typing

from fate.components.core.essential import DataframeArtifactType

from .._base_type import (
    ArtifactDescribe,
    DataOutputMetadata,
    Metadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import URI
    from fate.arch.dataframe import DataFrame

logger = logging.getLogger(__name__)


class DataframeWriter(_ArtifactTypeWriter[DataOutputMetadata]):
    def write(self, df: "DataFrame", name=None, namespace=None):
        self.artifact.consumed()
        logger.debug(f"start writing dataframe to artifact: {self.artifact}, name={name}, namespace={namespace}")
        from fate.arch import dataframe

        if name is not None:
            self.artifact.metadata.name = name
        if namespace is not None:
            self.artifact.metadata.namespace = namespace

        table = dataframe.serialize(self.ctx, df)
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

        logger.debug(f"write dataframe to artifact: <uri={self.artifact.uri}, type_name={self.artifact.type_name}>")


class DataframeReader(_ArtifactTypeReader):
    def read(self) -> "DataFrame":
        self.artifact.consumed()
        logger.debug(f"start reading dataframe from artifact: {self.artifact}")
        # if self.artifact.uri.scheme == "file":
        #     import inspect
        #
        #     from fate.arch import dataframe
        #
        #     kwargs = {}
        #     p = inspect.signature(dataframe.CSVReader.__init__).parameters
        #     parameter_keys = p.keys()
        #     for k, v in self.artifact.metadata.metadata.items():
        #         if k in parameter_keys:
        #             kwargs[k] = v
        #
        #     return dataframe.CSVReader(**kwargs).to_frame(self.ctx, self.artifact.uri.path)

        from fate.arch import dataframe

        table = self.ctx.computing.load(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata.get("schema", None),
            options=self.artifact.metadata.metadata.get("options", None),
        )
        df = dataframe.deserialize(self.ctx, table)
        logger.debug(f"read dataframe from artifact: <uri={self.artifact.uri}, type_name={self.artifact.type_name}>")
        return df


class DataframeArtifactDescribe(ArtifactDescribe[DataframeArtifactType, DataOutputMetadata]):
    @classmethod
    def get_type(cls):
        return DataframeArtifactType

    def get_writer(self, config, ctx, uri: "URI", type_name: str) -> DataframeWriter:
        from fate.components.core.spec.artifact import DataOutputMetadata

        return DataframeWriter(ctx, _ArtifactType(uri=uri, metadata=DataOutputMetadata(), type_name=type_name))

    def get_reader(self, ctx, uri: "URI", metadata: "Metadata", type_name: str) -> DataframeReader:
        return DataframeReader(ctx, _ArtifactType(uri, metadata=metadata, type_name=type_name))
