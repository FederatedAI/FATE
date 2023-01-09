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
import uuid


class OutputPool:
    def __init__(self, data, model, metric) -> None:
        self.data = data
        self.model = model
        self.metric = metric

    def create_data_artifact(self, name: str):
        return self.data.create_artifact(name)

    def create_model_artifact(self, name: str):
        return self.model.create_artifact(name)

    def create_metric_artifact(self, name: str):
        return self.metric.create_artifact(name)


def load_pool(output_pool_conf):
    data = _load_data_pool(output_pool_conf.data)
    model = _load_model_pool(output_pool_conf.model)
    metric = _load_metric_pool(output_pool_conf.metric)
    return OutputPool(data, model, metric)


def _load_data_pool(data_pool):
    from fate.arch.unify import URI
    from fate.components.spec.output import DirectoryDataPool

    if isinstance(data_pool, DirectoryDataPool):
        return DataPool(
            base_uri=URI.from_string(data_pool.metadata.uri).to_schema(),
            format=data_pool.metadata.format,
            name_template=data_pool.metadata.name_template,
        )
    raise RuntimeError(f"load data pool failed: {data_pool}")


def _load_model_pool(model_pool):
    from fate.arch.unify import URI
    from fate.components.spec.output import DirectoryModelPool

    if isinstance(model_pool, DirectoryModelPool):
        return ModelPool(
            base_uri=URI.from_string(model_pool.metadata.uri).to_schema(),
            format=model_pool.metadata.format,
            name_template=model_pool.metadata.name_template,
        )
    raise RuntimeError(f"load data pool failed: {model_pool}")


def _load_metric_pool(metric_pool):
    from fate.arch.unify import URI
    from fate.components.spec.output import DirectoryMetricPool

    if isinstance(metric_pool, DirectoryMetricPool):
        return MetricPool(
            base_uri=URI.from_string(metric_pool.metadata.uri).to_schema(),
            format=metric_pool.metadata.format,
            name_template=metric_pool.metadata.name_template,
        )
    raise RuntimeError(f"load data pool failed: {metric_pool}")


class DataPool:
    def __init__(self, base_uri, format, name_template) -> None:
        self.format = format
        self.base_uri = base_uri
        self.name_template = name_template

    def create_artifact(self, name):
        from fate.components import DatasetArtifact

        file_name = self.name_template.format(name=name, uuid=uuid.uuid1())
        uri = self.base_uri.create_file(file_name)
        metadata = dict(format=self.format)
        return DatasetArtifact(name=name, uri=uri.to_string(), metadata=metadata)


class ModelPool:
    def __init__(self, base_uri, format, name_template) -> None:
        self.format = format
        self.base_uri = base_uri
        self.name_template = name_template

    def create_artifact(self, name):
        from fate.components import ModelArtifact

        file_name = self.name_template.format(name=name, uuid=uuid.uuid1())
        uri = self.base_uri.create_file(file_name)
        metadata = dict(format=self.format)
        return ModelArtifact(name=name, uri=uri.to_string(), metadata=metadata)


class MetricPool:
    def __init__(self, base_uri, format, name_template) -> None:
        self.format = format
        self.base_uri = base_uri
        self.name_template = name_template

    def create_artifact(self, name):
        from fate.components import MetricArtifact

        file_name = self.name_template.format(name=name, uuid=uuid.uuid1())
        uri = self.base_uri.create_file(file_name)
        metadata = dict(format=self.format)
        return MetricArtifact(name=name, uri=uri.to_string(), metadata=metadata)
