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
import pickle

from arch.api import RuntimeInstance
from arch.api import WorkMode
from arch.api import session
from arch.api.utils import file_utils, version_control
from arch.api.proto import default_empty_fill_pb2
from fate_flow.settings import stat_logger


def save_component_model(component_model_key, model_buffers, party_model_id, model_version, version_log=None):
    """
    pipeline_model_table = session.table(name=model_version, namespace=party_model_id,
                                         partition=get_model_table_partition_count(),
                                         create_if_missing=True, error_if_exist=False, use_serialize=False)
    """
    pipeline_model_table = ModelTable(name=model_version, namespace=party_model_id)
    model_class_map = {}
    for buffer_name, buffer_object in model_buffers.items():
        storage_key = '{}:{}'.format(component_model_key, buffer_name)
        buffer_object_serialize_string = buffer_object.SerializeToString()
        if not buffer_object_serialize_string:
            fill_message = default_empty_fill_pb2.DefaultEmptyFillMessage()
            fill_message.flag = 'set'
            buffer_object_serialize_string = fill_message.SerializeToString()
        pipeline_model_table.put(storage_key, buffer_object_serialize_string, use_serialize=False)
        model_class_map[storage_key] = type(buffer_object).__name__
    # session.save_data_table_meta(model_class_map, data_table_namespace=party_model_id, data_table_name=model_version)
    pipeline_model_table.save_meta(model_class_map)
    # version_log = "[AUTO] save model at %s." % datetime.datetime.now() if not version_log else version_log
    # version_control.save_version(name=model_version, namespace=party_model_id, version_log=version_log)


def read_component_model(component_model_key, party_model_id, model_version):
    """
    pipeline_model_table = session.table(name=model_version, namespace=party_model_id,
                                         partition=get_model_table_partition_count(),
                                         create_if_missing=False, error_if_exist=False, use_serialize=False)
    """
    pipeline_model_table = ModelTable(name=model_version, namespace=party_model_id)
    model_buffers = {}
    if pipeline_model_table:
        model_class_map = pipeline_model_table.get_metas()
        for storage_key, buffer_object_bytes in pipeline_model_table.collect(use_serialize=False).items():
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
                parse_proto_object(proto_object=buffer_object, proto_object_serialized_bytes=buffer_object_bytes)
                model_buffers[buffer_name] = buffer_object
    return model_buffers


class ModelTable(object):
    def __init__(self, name, namespace):
        self.table_dir = os.path.join('./mlmodel/', namespace, name)
        self.table_meta_dir = os.path.join('./mlmodel/', namespace, '{}.meta'.format(name))
        if not os.path.exists(self.table_dir):
            os.makedirs(self.table_dir)
        if not os.path.exists(self.table_meta_dir):
            os.makedirs(self.table_meta_dir)

    def put(self, k, v, use_serialize=False):
        with open(os.path.join(self.table_dir, k), 'wb') as fw:
            pickle.dump(v, fw)

    def get(self, k):
        try:
            with open(os.path.join(self.table_dir, k), 'rb') as fr:
                return pickle.load(fr)
        except Exception as e:
            print(e)
            return None

    def collect(self, use_serialize=False):
        try:
            data = dict()
            for k in os.listdir(self.table_dir):
                with open(os.path.join(self.table_dir, k), 'rb') as fr:
                    data[k] = pickle.load(fr)
            return data
        except Exception as e:
            print(e)
            return {}

    def save_meta(self, meta: dict):
        for k, v in meta.items():
            with open(os.path.join(self.table_meta_dir, k), 'wb') as fw:
                pickle.dump(v, fw)

    def get_metas(self):
        try:
            meta = dict()
            for k in os.listdir(self.table_meta_dir):
                with open(os.path.join(self.table_meta_dir, k), 'rb') as fr:
                    meta[k] = pickle.load(fr)
            return meta
        except Exception as e:
            print(e)
            return None


def parse_proto_object(proto_object, proto_object_serialized_bytes):
    try:
        proto_object.ParseFromString(proto_object_serialized_bytes)
        stat_logger.info('parse {} proto object normal'.format(type(proto_object).__name__))
    except Exception as e1:
        try:
            fill_message = default_empty_fill_pb2.DefaultEmptyFillMessage()
            fill_message.ParseFromString(proto_object_serialized_bytes)
            proto_object.ParseFromString(bytes())
            stat_logger.info('parse {} proto object with default values'.format(type(proto_object).__name__))
        except Exception as e2:
            stat_logger.exception(e2)
            raise e1


def collect_pipeline_model(party_model_id, model_version):
    pipeline_model_table = session.table(name=model_version, namespace=party_model_id,
                                         partition=get_model_table_partition_count(),
                                         create_if_missing=False, error_if_exist=False)
    model_buffers = {}
    if pipeline_model_table:
        model_class_map = pipeline_model_table.get_metas()
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
    session.save_data_table_meta(kv, data_table_namespace=party_model_id, data_table_name=model_version)


def get_pipeline_model_meta(party_model_id, model_version):
    return session.get_data_table_metas(data_table_namespace=party_model_id, data_table_name=model_version)


def get_proto_buffer_class(class_name):
    package_path = os.path.join(file_utils.get_project_base_directory(), 'federatedml', 'protobuf', 'generated')
    package_python_path = 'federatedml.protobuf.generated'
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
