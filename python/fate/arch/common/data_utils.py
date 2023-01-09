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
import os
import uuid

from ..storage import StorageEngine
from .file_utils import get_project_base_directory


def default_output_info(task_id, task_version, output_type):
    return f"output_{output_type}_{task_id}_{task_version}", uuid.uuid1().hex


def default_input_fs_path(name, namespace, prefix=None, storage_engine=StorageEngine.HDFS):
    if storage_engine == StorageEngine.HDFS:
        return default_hdfs_path(data_type="input", name=name, namespace=namespace, prefix=prefix)
    elif storage_engine == StorageEngine.LOCALFS:
        return default_localfs_path(data_type="input", name=name, namespace=namespace)


def default_output_fs_path(name, namespace, prefix=None, storage_engine=StorageEngine.HDFS):
    if storage_engine == StorageEngine.HDFS:
        return default_hdfs_path(data_type="output", name=name, namespace=namespace, prefix=prefix)
    elif storage_engine == StorageEngine.LOCALFS:
        return default_localfs_path(data_type="output", name=name, namespace=namespace)


def default_localfs_path(name, namespace, data_type):
    return os.path.join(get_project_base_directory(), "localfs", data_type, namespace, name)


def default_hdfs_path(data_type, name, namespace, prefix=None):
    p = f"/fate/{data_type}_data/{namespace}/{name}"
    if prefix:
        p = f"{prefix}/{p}"
    return p
