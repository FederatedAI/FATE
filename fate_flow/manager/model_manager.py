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
import datetime
import importlib
import inspect
import os

from arch.api import RuntimeInstance
from arch.api import WorkMode
from arch.api.utils import file_utils
from fate_flow.manager import version_control
from fate_flow.settings import stat_logger
from fate_flow.storage.fate_storage import FateStorage


def save_component_model(component_model_key, model_buffers, party_model_id, model_version, version_log=None):
    pipeline_model_table = FateStorage.table(name=model_version, namespace=party_model_id,
                                             partition=get_model_table_partition_count(),
                                             create_if_missing=True, error_if_exist=False)
    model_class_map = {}
    for buffer_name, buffer_object in model_buffers.items():
        storage_key = '{}:{}'.format(component_model_key, buffer_name)
        pipeline_model_table.put(storage_key, buffer_object.SerializeToString(), use_serialize=False)
        model_class_map[storage_key] = type(buffer_object).__name__
    FateStorage.save_data_table_meta(model_class_map, namespace=party_model_id, name=model_version)
    version_log = "[AUTO] save model at %s." % datetime.datetime.now() if not version_log else version_log
    version_control.save_version(name=model_version, namespace=party_model_id, version_log=version_log)


def read_component_model(component_model_key, party_model_id, model_version):
    pipeline_model_table = FateStorage.table(name=model_version, namespace=party_model_id,
                                             partition=get_model_table_partition_count(),
                                             create_if_missing=False, error_if_exist=False)
    model_buffers = {}
    if pipeline_model_table:
        model_class_map = FateStorage.get_data_table_meta_by_instance(data_table=pipeline_model_table)
        for storage_key, buffer_object_bytes in pipeline_model_table.collect(use_serialize=False):
            storage_key_items = storage_key.split(':')
            buffer_name = ':'.join(storage_key_items[1:])
            current_model_key = storage_key_items[0]
            if current_model_key == component_model_key:
                buffer_object_class = get_proto_buffer_class(model_class_map.get(storage_key, ''))
                if buffer_object_class:
                    buffer_object = buffer_object_class()
                else:
                    raise Exception(
                        'can not found this protobuffer class: {}'.format(model_class_map.get(storage_key, '')))
                buffer_object.ParseFromString(buffer_object_bytes)
                model_buffers[buffer_name] = buffer_object
    return model_buffers


def collect_pipeline_model(party_model_id, model_version):
    pipeline_model_table = FateStorage.table(name=model_version, namespace=party_model_id,
                                             partition=get_model_table_partition_count(),
                                             create_if_missing=False, error_if_exist=False)
    model_buffers = {}
    if pipeline_model_table:
        model_class_map = FateStorage.get_data_table_meta_by_instance(data_table=pipeline_model_table)
        for storage_key, buffer_object_bytes in pipeline_model_table.collect(use_serialize=False):
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


def save_pipeline_model_meta(kv, party_model_id, model_version):
    FateStorage.save_data_table_meta(kv, namespace=party_model_id, name=model_version)


def get_pipeline_model_meta(party_model_id, model_version):
    return FateStorage.get_data_table_meta(namespace=party_model_id, name=model_version)


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
