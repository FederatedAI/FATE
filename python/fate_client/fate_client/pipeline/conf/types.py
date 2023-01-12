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
class JobStage(object):
    TRAIN = "train"
    PREDICT = "predict"
    DEFAULT = "default"


class ArtifactSourceType(object):
    TASK_OUTPUT_ARTIFACT = "task_output_artifact"
    MODEL_WAREHOUSE = "model_warehouse"


class ArtifactType(object):
    DATASET = "dataset"
    DATASETS = "datasets"
    MODEL = "model"
    MODELS = "models"
    METRIC = "metric"


class InputDataKeyType(object):
    TRAIN_DATA = "train_data"
    VALIDATE_DATA = "validate_data"
    TEST_DATA = "test_data"
    INPUT_DATA = "input_data"


class OutputDataKeyType(object):
    TRAIN_OUTPUT_DATA = "train_output_data"
    VALIDATE_OUTPUT_DATA = "validate_output_data"
    TEST_OUTPUT_DATA = "test_output_data"
    OUTPUT_DATA = "output_data"


class PlaceHolder(object):
    ...


class UriTypes(object):
    LOCAL = "file"
    SQL = "sql"
    LMDB = "lmdb"


class SupportRole(object):
    LOCAL = "local"
    GUEST = "guest"
    HOST = "host"
    ARBITER = "arbiter"

    @classmethod
    def support_roles(cls):
        return [cls.LOCAL, cls.GUEST, cls.HOST, cls.ARBITER]
