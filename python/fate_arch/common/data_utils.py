import os
import uuid

from fate_arch.common import file_utils
from fate_arch.storage import StorageEngine


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
    return os.path.join(file_utils.get_project_base_directory(), 'localfs', data_type, namespace, name)


def default_hdfs_path(data_type, name, namespace, prefix=None):
    p = f"/fate/{data_type}_data/{namespace}/{name}"
    if prefix:
        p = f"{prefix}/{p}"
    return p
