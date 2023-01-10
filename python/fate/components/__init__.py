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
from typing import Dict, List, Literal, Optional, Type, TypeVar

from typing_extensions import Annotated

T_ROLE = Literal["guest", "host", "arbiter"]
T_STAGE = Literal["train", "predict", "default"]
T_LABEL = Literal["trainable"]


class Role:
    def __init__(self, name: T_ROLE) -> None:
        self.name: T_ROLE = name

    @property
    def is_guest(self) -> bool:
        return self.name == "guest"

    @property
    def is_host(self) -> bool:
        return self.name == "host"

    @property
    def is_arbiter(self) -> bool:
        return self.name == "arbiter"


GUEST = Role("guest")
HOST = Role("host")
ARBITER = Role("arbiter")

T_ROLE = Literal["guest", "host", "arbiter"]
T_STAGE = Literal["train", "predict", "default"]
T_LABEL = Literal["trainable"]


class Stage:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_train(self):
        return self.name == "train"

    @property
    def is_predict(self):
        return self.name == "predict"

    @property
    def is_default(self):
        return self.name == "default"


TRAIN = Stage("train")
PREDICT = Stage("predict")
DEFAULT = Stage("default")


class LABELS:
    TRAINABLE = "trainable"


class OutputAnnotated:
    ...


class InputAnnotated:
    ...


T = TypeVar("T")
Output = Annotated[T, OutputAnnotated]
Input = Annotated[T, InputAnnotated]


class Artifact:
    type: str = "artifact"
    """Represents a generic machine learning artifact.

    This class and all artifact classes
    store the name, uri, and metadata for a machine learning artifact.
    Use this artifact type when an artifact
    does not fit into another more specific artifact type (e.g., ``Model``, ``Dataset``).

    Args:
        name: Name of the artifact.
        uri: The artifact's location on disk or cloud storage.
        metadata: Arbitrary key-value pairs about the artifact.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Initializes the Artifact with the given name, URI and metadata."""
        self.uri = uri or ""
        self.name = name or ""
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return f"<{type(self).__name__} {dict(name=self.name, uri=self.uri, metadata=self.metadata)}>"

    def __repr__(self) -> str:
        return self.__str__()


class Artifacts:
    type: str
    artifact_type: Type[Artifact]

    def __init__(self, artifacts: List[Artifact]) -> None:
        self.artifacts = artifacts

    def __str__(self) -> str:
        return f"<{type(self).__name__} {self.artifacts}>"

    def __repr__(self) -> str:
        return self.__str__()


class DatasetArtifact(Artifact):
    type = "dataset"
    """An artifact representing a machine learning dataset.

    Args:
        name: Name of the dataset.
        uri: The dataset's location on disk or cloud storage.
        metadata: Arbitrary key-value pairs about the dataset.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)


class DatasetArtifacts(Artifacts):
    type = "datasets"
    artifact_type: Type[Artifact] = DatasetArtifact


class ModelArtifact(Artifact):
    type = "model"
    """An artifact representing a machine learning model.

    Args:
        name: Name of the model.
        uri: The model's location on disk or cloud storage.
        metadata: Arbitrary key-value pairs about the model.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)


class ModelArtifacts(Artifacts):
    type = "models"


class MetricArtifact(Artifact):
    type = "metric"

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)


class LossMetrics(MetricArtifact):
    type = "loss"

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)


class ClassificationMetrics(MetricArtifact):
    """An artifact for storing classification metrics.

    Args:
        name: Name of the metrics artifact.
        uri: The metrics artifact's location on disk or cloud storage.
        metadata: The key-value scalar metrics.
    """

    type = "classification_metrics"

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(uri=uri, name=name, metadata=metadata)
