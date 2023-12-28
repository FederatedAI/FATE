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

from typing import List, Optional, Type

from .._base_type import Role, _create_artifact_annotation
from ._json import (
    JsonMetricArtifactDescribe,
    JsonMetricFileWriter,
    JsonMetricRestfulWriter,
)


def json_metric_output(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[JsonMetricFileWriter]:
    return _create_artifact_annotation(False, False, JsonMetricArtifactDescribe, "metric")(roles, desc, optional)


def json_metric_outputs(roles: Optional[List[Role]] = None, desc="", optional=False) -> Type[JsonMetricFileWriter]:
    return _create_artifact_annotation(False, True, JsonMetricArtifactDescribe, "metric")(roles, desc, optional)
