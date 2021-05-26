#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
#

from enum import Enum

from .minio import MinIOModelStorage


class ModelStorageType(Enum):
    MINIO = 1


def get_model_storage(storage_type=ModelStorageType.MINIO, **kwargs):
    """
    get a model storage insance based on the specified type

    :param storage_type: type of the model storage, currently only MINIO is supported
    :param kwargs: keyword arguments to initialize the model storage object
    :return: an instance of a subclass of BaseModelStorage
    """
    if isinstance(storage_type, str):
        storage_type = ModelStorageType[storage_type.upper()]
    if storage_type == ModelStorageType.MINIO:
        return MinIOModelStorage(**kwargs)
    else:
        raise ValueError("unknown model storage type: {}".format(storage_type))
