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

import typing

from fate.components.core.essential import TableArtifactType

from .._base_type import (
    URI,
    ArtifactDescribe,
    DataOutputMetadata,
    Metadata,
    _ArtifactType,
    _ArtifactTypeReader,
    _ArtifactTypeWriter,
)

if typing.TYPE_CHECKING:
    from fate.arch import Context


class TableWriter(_ArtifactTypeWriter[DataOutputMetadata]):
    def write(self, table):
        self.artifact.consumed()
        if "schema" not in self.artifact.metadata.metadata:
            self.artifact.metadata.metadata["schema"] = {}
        table.save(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata["schema"],
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableReader(_ArtifactTypeReader):
    def read(self):
        self.artifact.consumed()
        return self.ctx.computing.load(
            uri=self.artifact.uri,
            schema=self.artifact.metadata.metadata.get("schema", {}),
            options=self.artifact.metadata.metadata.get("options", None),
        )


class TableArtifactDescribe(ArtifactDescribe[TableArtifactType, DataOutputMetadata]):
    @classmethod
    def get_type(cls):
        return TableArtifactType

    def get_writer(self, config, ctx: "Context", uri: URI, type_name: str) -> TableWriter:
        return TableWriter(ctx, _ArtifactType(uri, DataOutputMetadata(), type_name))

    def get_reader(self, ctx: "Context", uri: "URI", metadata: "Metadata", type_name: str) -> TableReader:
        return TableReader(ctx, _ArtifactType(uri, metadata, type_name))
