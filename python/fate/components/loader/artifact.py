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
def load_artifact(data, artifact_type):
    from fate.components import (
        Artifact,
        Artifacts,
        DatasetArtifact,
        DatasetArtifacts,
        MetricArtifact,
        ModelArtifact,
        ModelArtifacts,
    )

    if hasattr(artifact_type, "__origin__"):
        artifact_type = artifact_type.__origin__
    if isinstance(data, list):
        if artifact_type.__origin__ == DatasetArtifacts:
            return DatasetArtifacts([DatasetArtifact(name=d.name, uri=d.uri, metadata=d.metadata) for d in data])
        if artifact_type == ModelArtifacts:
            return ModelArtifacts([ModelArtifact(name=d.name, uri=d.uri, metadata=d.metadata) for d in data])
        return Artifacts([Artifact(name=d.name, uri=d.uri, metadata=d.metadata) for d in data])
    else:
        if artifact_type == DatasetArtifact:
            return DatasetArtifact(name=data.name, uri=data.uri, metadata=data.metadata)
        if artifact_type == ModelArtifact:
            return ModelArtifact(name=data.name, uri=data.uri, metadata=data.metadata)
        if artifact_type == MetricArtifact:
            return MetricArtifact(name=data.name, uri=data.uri, metadata=data.metadata)
        return Artifact(name=data.name, uri=data.uri, metadata=data.metadata)
