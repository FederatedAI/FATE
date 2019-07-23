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
#
import os
from arch.api.proto.model_meta_pb2 import ModelMeta
from arch.api.proto.model_param_pb2 import ModelParam
from arch.api.proto.data_transform_server_pb2 import DataTransformServer
from arch.api.utils.core import json_loads
from arch.api.utils.format_transform import camel_to_pascal
from fate_flow.storage.fate_storage import FateStorage
from arch.api import RuntimeInstance
from arch.api import WorkMode
from fate_flow.manager import version_control
import datetime
import inspect
import importlib
from fate_flow.settings import stat_logger
from arch.api.utils import file_utils


def save_model(model_key, model_buffers, model_version, model_id, version_log=None):
    data_table = FateStorage.table(name=model_version, namespace=model_id, partition=get_model_table_partition_count(),
                                   create_if_missing=True, error_if_exist=False)
    model_class_map = {}
    for buffer_name, buffer_object in model_buffers.items():
        storage_key = '{}.{}'.format(model_key, buffer_name)
        data_table.put(storage_key, buffer_object.SerializeToString(), use_serialize=False)
        model_class_map[storage_key] = type(buffer_object).__name__
    FateStorage.save_data_table_meta(model_class_map, namespace=model_id, name=model_version)
    version_log = "[AUTO] save model at %s." % datetime.datetime.now() if not version_log else version_log
    version_control.save_version(name=model_version, namespace=model_id, version_log=version_log)


def read_model(model_key, model_version, model_id):
    data_table = FateStorage.table(name=model_version, namespace=model_id, partition=get_model_table_partition_count(),
                                   create_if_missing=False, error_if_exist=False)
    model_buffers = {}
    if data_table:
        model_class_map = FateStorage.get_data_table_meta_by_instance(data_table=data_table)
        for storage_key, buffer_object_bytes in data_table.collect(use_serialize=False):
            storage_key_items = storage_key.split('.')
            buffer_name = '.'.join(storage_key_items[1:])
            current_model_key = storage_key_items[0]
            if current_model_key == model_key:
                buffer_object_class = get_proto_buffer_class(model_class_map.get(storage_key, ''))
                if buffer_object_class:
                    buffer_object = buffer_object_class()
                else:
                    raise Exception('can not found this protobuffer class: {}'.format(model_class_map.get(storage_key, '')))
                buffer_object.ParseFromString(buffer_object_bytes)
                model_buffers[buffer_name] = buffer_object
    return model_buffers


def collect_model(model_version, model_id):
    data_table = FateStorage.table(name=model_version, namespace=model_id, partition=get_model_table_partition_count(),
                                   create_if_missing=False, error_if_exist=False)
    model_buffers = {}
    if data_table:
        model_class_map = FateStorage.get_data_table_meta_by_instance(data_table=data_table)
        for storage_key, buffer_object_bytes in data_table.collect(use_serialize=False):
            storage_key_items = storage_key.split('.')
            buffer_name = storage_key_items[-1]
            buffer_object_class = get_proto_buffer_class(model_class_map.get(storage_key, ''))
            if buffer_object_class:
                buffer_object = buffer_object_class()
            else:
                raise Exception('can not found this protobuffer class: {}'.format(model_class_map.get(storage_key, '')))
            buffer_object.ParseFromString(buffer_object_bytes)
            model_buffers[buffer_name] = buffer_object
    return model_buffers


def save_model_meta(kv, model_version, model_id):
    FateStorage.save_data_table_meta(kv, namespace=model_id, name=model_version)


def get_model_meta(model_version, model_id):
    return FateStorage.get_data_table_meta(namespace=model_id, name=model_version)


def get_proto_buffer_class(class_name):
    package_path = os.path.join(file_utils.get_project_base_directory(), 'arch', 'api', 'proto')
    package_python_path = 'arch.api.proto'
    for f in os.listdir(package_path):
        if f.startswith('.'):
            continue
        try:
            proto_module = importlib.import_module(package_python_path + '.' + f.rstrip('.py'))
            for name, obj in inspect.getmembers(proto_module):
                if inspect.isclass(obj) and name == class_name:
                    return obj
        except Exception as e:
            stat_logger.warning(e)
    else:
        return None


def get_model_table_partition_count():
    # todo: max size limit?
    return 4 if RuntimeInstance.MODE == WorkMode.CLUSTER else 1


def test_model(role):
    with open("%s_runtime_conf.json" % role) as conf_fr:
        runtime_conf = json_loads(conf_fr.read())

    model_table_name = runtime_conf.get("WorkFlowParam").get("model_table")
    model_table_namespace = runtime_conf.get("WorkFlowParam").get("model_namespace")
    print(model_table_name, model_table_namespace)
    model_meta_save = ModelMeta()
    model_meta_save.name = "HeteroLR%s" % (camel_to_pascal(role))
    save_model("model_meta", model_meta_save, model_version=model_table_name, model_id=model_table_namespace)

    model_meta_read = ModelMeta()
    read_model("model_meta", model_meta_read, model_version=model_table_name, model_id=model_table_namespace)
    print(model_meta_read)

    model_param_save = ModelParam()
    model_param_save.weight["k1"] = 1
    model_param_save.weight["k2"] = 2
    save_model("model_param", model_param_save, model_version=model_table_name, model_id=model_table_namespace)

    # read
    model_param_read = ModelParam()
    read_model("model_param", model_param_read, model_version=model_table_name, model_id=model_table_namespace)
    print(model_param_read)

    data_transform = DataTransformServer()
    data_transform.missing_replace_method = "xxxx"
    save_model("data_transform", data_transform, model_version=model_table_name, model_id=model_table_namespace)


if __name__ == '__main__':
    import uuid

    job_id = str(uuid.uuid1().hex)
    FateStorage.init_storage(job_id=job_id)

    test_model("guest")
    test_model("host")

    print(job_id)
