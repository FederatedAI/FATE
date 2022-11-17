# Copyright 2021 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes for input/output Artifacts in KFP SDK.

These are only compatible with v2 Pipelines.
"""

from typing import Dict, List, Optional, Type

from typing_extensions import Annotated

from .types import Input, Output


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

    @classmethod
    def parse_desc(cls, desc):
        return cls(uri=desc.uri, name=desc.name, metadata=desc.metadata)

    @property
    def path(self) -> str:
        return self._get_path()

    @path.setter
    def path(self, path: str) -> None:
        self._set_path(path)

    def _get_path(self) -> Optional[str]:
        if self.uri.startswith("gs://"):
            return _GCS_LOCAL_MOUNT_PREFIX + self.uri[len("gs://") :]
        elif self.uri.startswith("minio://"):
            return _MINIO_LOCAL_MOUNT_PREFIX + self.uri[len("minio://") :]
        elif self.uri.startswith("s3://"):
            return _S3_LOCAL_MOUNT_PREFIX + self.uri[len("s3://") :]
        return None

    def _set_path(self, path: str) -> None:
        if path.startswith(_GCS_LOCAL_MOUNT_PREFIX):
            path = "gs://" + path[len(_GCS_LOCAL_MOUNT_PREFIX) :]
        elif path.startswith(_MINIO_LOCAL_MOUNT_PREFIX):
            path = "minio://" + path[len(_MINIO_LOCAL_MOUNT_PREFIX) :]
        elif path.startswith(_S3_LOCAL_MOUNT_PREFIX):
            path = "s3://" + path[len(_S3_LOCAL_MOUNT_PREFIX) :]
        self.uri = path


class Artifacts:
    type: str
    artifact_type: Type[Artifact]

    def __init__(self, artifacts: List[Artifact]) -> None:
        self.artifacts = artifacts

    @classmethod
    def parse_desc(cls, desc_list):
        artifacts = []
        for desc in desc_list:
            artifacts.append(cls.artifact_type(uri=desc.uri, name=desc.name, metadata=desc.metadata))
        return cls(artifacts)


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

    @property
    def framework(self) -> str:
        return self._get_framework()

    def _get_framework(self) -> str:
        return self.metadata.get("framework", "")

    @framework.setter
    def framework(self, framework: str) -> None:
        self._set_framework(framework)

    def _set_framework(self, framework: str) -> None:
        self.metadata["framework"] = framework


class ModelArtifacts(Artifacts):
    type = "models"
    artifact_type: Type[Artifact] = ModelArtifact


class MetricArtifact(Artifact):
    type = "metric"

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)

    def log_metric(self, metric: str, value: float) -> None:
        """Sets a custom scalar metric in the artifact's metadata.

        Args:
          metric: The metric key.
          value: The metric value.
        """
        self.metadata[metric] = value


class ClassificationMetrics(Artifact):
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

    def log_roc_data_point(self, fpr: float, tpr: float, threshold: float) -> None:
        """Logs a single data point in the ROC curve to metadata.

        Args:
          fpr: False positive rate value of the data point.
          tpr: True positive rate value of the data point.
          threshold: Threshold value for the data point.
        """

        roc_reading = {
            "confidenceThreshold": threshold,
            "recall": tpr,
            "falsePositiveRate": fpr,
        }
        if "confidenceMetrics" not in self.metadata.keys():
            self.metadata["confidenceMetrics"] = []

        self.metadata["confidenceMetrics"].append(roc_reading)

    def log_roc_curve(self, fpr: List[float], tpr: List[float], threshold: List[float]) -> None:
        """Logs an ROC curve to metadata.

        Args:
          fpr: List of false positive rate values.
          tpr: List of true positive rate values.
          threshold: List of threshold values.

        Raises:
          ValueError: If the lists ``fpr``, ``tpr`` and ``threshold`` are not the same length.
        """
        if len(fpr) != len(tpr) or len(fpr) != len(threshold) or len(tpr) != len(threshold):
            raise ValueError(
                "Length of fpr, tpr and threshold must be the same. "
                "Got lengths {}, {} and {} respectively.".format(len(fpr), len(tpr), len(threshold))
            )

        for i in range(len(fpr)):
            self.log_roc_data_point(fpr=fpr[i], tpr=tpr[i], threshold=threshold[i])

    def set_confusion_matrix_categories(self, categories: List[str]) -> None:
        """Stores confusion matrix categories to metadata.

        Args:
          categories: List of strings specifying the categories.
        """

        self._categories = []
        annotation_specs = []
        for category in categories:
            annotation_spec = {"displayName": category}
            self._categories.append(category)
            annotation_specs.append(annotation_spec)

        self._matrix = []
        for row in range(len(self._categories)):
            self._matrix.append({"row": [0] * len(self._categories)})

        self._confusion_matrix = {}
        self._confusion_matrix["annotationSpecs"] = annotation_specs
        self._confusion_matrix["rows"] = self._matrix
        self.metadata["confusionMatrix"] = self._confusion_matrix

    def log_confusion_matrix_row(self, row_category: str, row: List[float]) -> None:
        """Logs a confusion matrix row to metadata.

        Args:
          row_category: Category to which the row belongs.
          row: List of integers specifying the values for the row.

        Raises:
          ValueError: If ``row_category`` is not in the list of categories
            set in ``set_categories`` call.
        """
        if row_category not in self._categories:
            raise ValueError("Invalid category: {} passed. Expected one of: {}".format(row_category, self._categories))

        if len(row) != len(self._categories):
            raise ValueError("Invalid row. Expected size: {} got: {}".format(len(self._categories), len(row)))

        self._matrix[self._categories.index(row_category)] = {"row": row}
        self.metadata["confusionMatrix"] = self._confusion_matrix

    def log_confusion_matrix_cell(self, row_category: str, col_category: str, value: int) -> None:
        """Logs a cell in the confusion matrix to metadata.

        Args:
          row_category: String representing the name of the row category.
          col_category: String representing the name of the column category.
          value: Value of the cell.

        Raises:
          ValueError: If ``row_category`` or ``col_category`` is not in the list of
           categories set in ``set_categories``.
        """
        if row_category not in self._categories:
            raise ValueError("Invalid category: {} passed. Expected one of: {}".format(row_category, self._categories))

        if col_category not in self._categories:
            raise ValueError("Invalid category: {} passed. Expected one of: {}".format(row_category, self._categories))

        self._matrix[self._categories.index(row_category)]["row"][self._categories.index(col_category)] = value
        self.metadata["confusionMatrix"] = self._confusion_matrix

    def log_confusion_matrix(self, categories: List[str], matrix: List[List[int]]) -> None:
        """Logs a confusion matrix to metadata.

        Args:
          categories: List of the category names.
          matrix: Complete confusion matrix.

        Raises:
          ValueError: If the length of ``categories`` does not match number of rows or columns of ``matrix``.
        """
        self.set_confusion_matrix_categories(categories)

        if len(matrix) != len(categories):
            raise ValueError("Invalid matrix: {} passed for categories: {}".format(matrix, categories))

        for index in range(len(categories)):
            if len(matrix[index]) != len(categories):
                raise ValueError("Invalid matrix: {} passed for categories: {}".format(matrix, categories))

            self.log_confusion_matrix_row(categories[index], matrix[index])

        self.metadata["confusionMatrix"] = self._confusion_matrix


class SlicedClassificationMetrics(Artifact):
    """An artifact for storing sliced classification metrics.

    Similar to ``ClassificationMetrics``, tasks using this class are
    expected to use log methods of the class to log metrics with the
    difference being each log method takes a slice to associate the
    ``ClassificationMetrics``.

    Args:
        name: Name of the metrics artifact.
        uri: The metrics artifact's location on disk or cloud storage.
        metadata: Arbitrary key-value pairs about the metrics artifact.
    """

    type = "sliced_classification_metrics"

    def __init__(
        self,
        name: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(uri=uri, name=name, metadata=metadata)

    def _upsert_classification_metrics_for_slice(self, slice: str) -> None:
        """Upserts the classification metrics instance for a slice."""
        if slice not in self._sliced_metrics:
            self._sliced_metrics[slice] = ClassificationMetrics()

    def _update_metadata(self, slice: str) -> None:
        """Updates metadata to adhere to the metrics schema."""
        self.metadata = {}
        self.metadata["evaluationSlices"] = []
        for slice in self._sliced_metrics.keys():
            slice_metrics = {
                "slice": slice,
                "sliceClassificationMetrics": self._sliced_metrics[slice].metadata,
            }
            self.metadata["evaluationSlices"].append(slice_metrics)

    def log_roc_reading(self, slice: str, threshold: float, tpr: float, fpr: float) -> None:
        """Logs a single data point in the ROC curve of a slice to metadata.

        Args:
          slice: String representing slice label.
          threshold: Thresold value for the data point.
          tpr: True positive rate value of the data point.
          fpr: False positive rate value of the data point.
        """

        self._upsert_classification_metrics_for_slice(slice)
        self._sliced_metrics[slice].log_roc_reading(threshold, tpr, fpr)
        self._update_metadata(slice)

    def load_roc_readings(self, slice: str, readings: List[List[float]]) -> None:
        """Bulk loads ROC curve readings for a slice.

        Args:
          slice: String representing slice label.
          readings: A 2-dimensional list providing ROC curve data points. The expected order of the data points is: threshold, true positive rate, false positive rate.
        """
        self._upsert_classification_metrics_for_slice(slice)
        self._sliced_metrics[slice].load_roc_readings(readings)
        self._update_metadata(slice)

    def set_confusion_matrix_categories(self, slice: str, categories: List[str]) -> None:
        """Logs confusion matrix categories for a slice to metadata.

        Categories are stored in the internal ``metrics_utils.ConfusionMatrix``
        instance of the slice.

        Args:
          slice: String representing slice label.
          categories: List of strings specifying the categories.
        """
        self._upsert_classification_metrics_for_slice(slice)
        self._sliced_metrics[slice].set_confusion_matrix_categories(categories)
        self._update_metadata(slice)

    def log_confusion_matrix_row(self, slice: str, row_category: str, row: List[int]) -> None:
        """Logs a confusion matrix row for a slice to metadata.

        Row is updated on the internal ``metrics_utils.ConfusionMatrix``
        instance of the slice.

        Args:
          slice: String representing slice label.
          row_category: Category to which the row belongs.
          row: List of integers specifying the values for the row.
        """
        self._upsert_classification_metrics_for_slice(slice)
        self._sliced_metrics[slice].log_confusion_matrix_row(row_category, row)
        self._update_metadata(slice)

    def log_confusion_matrix_cell(self, slice: str, row_category: str, col_category: str, value: int) -> None:
        """Logs a confusion matrix cell for a slice to metadata.

        Cell is updated on the internal ``metrics_utils.ConfusionMatrix``
        instance of the slice.

        Args:
          slice: String representing slice label.
          row_category: String representing the name of the row category.
          col_category: String representing the name of the column category.
          value: Value of the cell.
        """
        self._upsert_classification_metrics_for_slice(slice)
        self._sliced_metrics[slice].log_confusion_matrix_cell(row_category, col_category, value)
        self._update_metadata(slice)

    def load_confusion_matrix(self, slice: str, categories: List[str], matrix: List[List[int]]) -> None:
        """Bulk loads the whole confusion matrix for a slice.

        Args:
          slice: String representing slice label.
          categories: List of the category names.
          matrix: Complete confusion matrix.
        """
        self._upsert_classification_metrics_for_slice(slice)
        self._sliced_metrics[slice].log_confusion_matrix_cell(categories, matrix)
        self._update_metadata(slice)


TrainData = Input[DatasetArtifact]
ValidateData = Input[DatasetArtifact]
TestData = Input[DatasetArtifact]
TrainOutputData = Output[DatasetArtifact]
TestOutputData = Output[DatasetArtifact]
Metrics = Output[MetricArtifact]
