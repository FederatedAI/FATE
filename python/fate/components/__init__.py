from typing import Literal, Type, TypeVar

from typing_extensions import Annotated

GUEST = "guest"
HOST = "host"
ARBITER = "arbiter"

T_ROLE = Literal["guest", "host", "arbiter"]
T_STAGE = Literal["train", "predict", "default"]
T_LABEL = Literal["trainable"]


class STAGES:
    TRAIN = "train"
    PREDICT = "predict"
    DEFAULT = "default"


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
    """


class Artifacts:
    type: str = "artifacts"


class DatasetArtifact(Artifact):
    type = "dataset"
    """An artifact representing a machine learning dataset.
    """


class DatasetArtifacts(Artifacts):
    type = "datasets"


class ModelArtifact(Artifact):
    type = "model"
    """An artifact representing a machine learning model.
    """


class ModelArtifacts(Artifacts):
    type = "models"
    artifact_type: Type[Artifact] = ModelArtifact


class MetricArtifact(Artifact):
    type = "metric"


class ClassificationMetrics(Artifact):
    """An artifact for storing classification metrics."""

    type = "classification_metrics"


class SlicedClassificationMetrics(Artifact):
    """An artifact for storing sliced classification metrics.

    Similar to ``ClassificationMetrics``, tasks using this class are
    expected to use log methods of the class to log metrics with the
    difference being each log method takes a slice to associate the
    ``ClassificationMetrics``.
    """

    type = "sliced_classification_metrics"
