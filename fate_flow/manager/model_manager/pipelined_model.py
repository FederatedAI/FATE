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
import importlib
import inspect
import os
import shutil
import base64
from ruamel import yaml

from arch.api.utils import file_utils
from arch.api.proto import default_empty_fill_pb2
from fate_flow.settings import stat_logger, TEMP_DIRECTORY


class PipelinedModel(object):
    def __init__(self, model_id, model_version):
        """
        Support operations on FATE PipelinedModels
        TODO: add lock
        :param model_id:
        :param model_version:
        """
        self.model_id = model_id
        self.model_version = model_version
        self.model_path = os.path.join(file_utils.get_project_base_directory(), "model_local_cache", model_id, model_version)
        self.define_proto_path = os.path.join(self.model_path, "define", "proto")
        self.define_meta_path = os.path.join(self.model_path, "define", "define_meta.yaml")
        self.variables_index_path = os.path.join(self.model_path, "variables", "index")
        self.variables_data_path = os.path.join(self.model_path, "variables", "data")
        self.default_archive_format = "zip"

    def create_pipelined_model(self):
        if os.path.exists(self.model_path):
            raise Exception("Model creation failed because it has already been created, model cache path is {}".format(
                self.model_path
            ))
        else:
            os.makedirs(self.model_path, exist_ok=False)
        for path in [self.variables_index_path, self.variables_data_path]:
            os.makedirs(path, exist_ok=False)
        shutil.copytree(os.path.join(file_utils.get_project_base_directory(), "federatedml", "protobuf", "proto"), self.define_proto_path)
        with open(self.define_meta_path, "w", encoding="utf-8") as fw:
            yaml.dump({"describe": "This is the model definition meta"}, fw, Dumper=yaml.RoundTripDumper)

    def save_component_model(self, component_name, component_module_name, model_alias, model_buffers):
        model_proto_index = {}
        component_model_storage_path = os.path.join(self.variables_data_path, component_name, model_alias)
        os.makedirs(component_model_storage_path, exist_ok=True)
        for model_name, buffer_object in model_buffers.items():
            storage_path = os.path.join(component_model_storage_path, model_name)
            buffer_object_serialized_string = buffer_object.SerializeToString()
            if not buffer_object_serialized_string:
                fill_message = default_empty_fill_pb2.DefaultEmptyFillMessage()
                fill_message.flag = 'set'
                buffer_object_serialized_string = fill_message.SerializeToString()
            with open(storage_path, "wb") as fw:
                fw.write(buffer_object_serialized_string)
            model_proto_index[model_name] = type(buffer_object).__name__   # index of model name and proto buffer class name
            stat_logger.info("Save {} {} {} buffer".format(component_name, model_alias, model_name))
        self.update_component_meta(component_name=component_name,
                                   component_module_name=component_module_name,
                                   model_alias=model_alias,
                                   model_proto_index=model_proto_index)
        stat_logger.info("Save {} {} successfully".format(component_name, model_alias))

    def read_component_model(self, component_name, model_alias):
        component_model_storage_path = os.path.join(self.variables_data_path, component_name, model_alias)
        model_proto_index = self.get_model_proto_index(component_name=component_name,
                                                       model_alias=model_alias)
        model_buffers = {}
        for model_name, buffer_name in model_proto_index.items():
            with open(os.path.join(component_model_storage_path, model_name), "rb") as fr:
                buffer_object_serialized_string = fr.read()
                model_buffers[model_name] = self.parse_proto_object(buffer_name=buffer_name,
                                                                    buffer_object_serialized_string=buffer_object_serialized_string)
        return model_buffers

    def collect_models(self, in_bytes=False, b64encode=True):
        model_buffers = {}
        with open(self.define_meta_path, "r", encoding="utf-8") as fr:
            define_index = yaml.safe_load(fr)
            for component_name in define_index.get("model_proto", {}).keys():
                for model_alias, model_proto_index in define_index["model_proto"][component_name].items():
                    component_model_storage_path = os.path.join(self.variables_data_path, component_name, model_alias)
                    for model_name, buffer_name in model_proto_index.items():
                        with open(os.path.join(component_model_storage_path, model_name), "rb") as fr:
                            buffer_object_serialized_string = fr.read()
                            if not in_bytes:
                                model_buffers[model_name] = self.parse_proto_object(buffer_name=buffer_name,
                                                                                    buffer_object_serialized_string=buffer_object_serialized_string)
                            else:
                                if b64encode:
                                    buffer_object_serialized_string = base64.b64encode(buffer_object_serialized_string).decode()
                                model_buffers["{}.{}:{}".format(component_name, model_alias, model_name)] = buffer_object_serialized_string
        return model_buffers

    def save_pipeline(self, pipelined_buffer_object):
        buffer_object_serialized_string = pipelined_buffer_object.SerializeToString()
        if not buffer_object_serialized_string:
            fill_message = default_empty_fill_pb2.DefaultEmptyFillMessage()
            fill_message.flag = 'set'
            buffer_object_serialized_string = fill_message.SerializeToString()
        with open(os.path.join(self.model_path, "pipeline.pb"), "wb") as fw:
            fw.write(buffer_object_serialized_string)

    def packaging_model(self):
        if not os.path.exists(self.model_path):
            raise Exception("Can not found {} {} model local cache".format(self.model_id, self.model_version))
        archive_file_path = shutil.make_archive(base_name=self.archive_model_base_path(), format=self.default_archive_format, root_dir=self.model_path)
        stat_logger.info("Make model {} {} archive on {} successfully".format(self.model_id,
                                                                              self.model_version,
                                                                              archive_file_path))
        return archive_file_path

    def unpack_model(self, archive_file_path: str):
        if os.path.exists(self.model_path):
            raise Exception("Model {} {} local cache already existed".format(self.model_id, self.model_version))
        shutil.unpack_archive(archive_file_path, self.model_path)
        stat_logger.info("Unpack model archive to {}".format(self.model_path))

    def update_component_meta(self, component_name, component_module_name, model_alias, model_proto_index):
        """
        update meta info yaml
        TODO: with lock
        :param component_name:
        :param component_module_name:
        :param model_alias:
        :param model_proto_index:
        :return:
        """
        with open(self.define_meta_path, "r", encoding="utf-8") as fr:
            define_index = yaml.safe_load(fr)
        with open(self.define_meta_path, "w", encoding="utf-8") as fw:
            define_index["component_define"] = define_index.get("component_define", {})
            define_index["component_define"][component_name] = define_index["component_define"].get(component_name, {})
            define_index["component_define"][component_name].update({"module_name": component_module_name})
            define_index["model_proto"] = define_index.get("model_proto", {})
            define_index["model_proto"][component_name] = define_index["model_proto"].get(component_name, {})
            define_index["model_proto"][component_name][model_alias] = define_index["model_proto"][component_name].get(model_alias, {})
            define_index["model_proto"][component_name][model_alias].update(model_proto_index)
            yaml.dump(define_index, fw, Dumper=yaml.RoundTripDumper)

    def get_model_proto_index(self, component_name, model_alias):
        with open(self.define_meta_path, "r", encoding="utf-8") as fr:
            define_index = yaml.safe_load(fr)
            return define_index.get("model_proto", {}).get(component_name, {}).get(model_alias, {})

    def get_component_define(self, component_name=None):
        with open(self.define_meta_path, "r", encoding="utf-8") as fr:
            define_index = yaml.safe_load(fr)
            if component_name:
                return define_index.get("component_define", {}).get(component_name, {})
            else:
                return define_index.get("component_define", {})

    def parse_proto_object(self, buffer_name, buffer_object_serialized_string):
        try:
            buffer_object = self.get_proto_buffer_class(buffer_name)()
        except Exception as e:
            stat_logger.exception("Can not restore proto buffer object", e)
            raise e
        try:
            buffer_object.ParseFromString(buffer_object_serialized_string)
            stat_logger.info('parse {} proto object normal'.format(type(buffer_object).__name__))
            return buffer_object
        except Exception as e1:
            try:
                fill_message = default_empty_fill_pb2.DefaultEmptyFillMessage()
                fill_message.ParseFromString(buffer_object_serialized_string)
                buffer_object.ParseFromString(bytes())
                stat_logger.info('parse {} proto object with default values'.format(type(buffer_object).__name__))
                return buffer_object
            except Exception as e2:
                stat_logger.exception(e2)
                raise e1

    @staticmethod
    def get_proto_buffer_class(buffer_name):
        package_path = os.path.join(file_utils.get_project_base_directory(), 'federatedml', 'protobuf', 'generated')
        package_python_path = 'federatedml.protobuf.generated'
        for f in os.listdir(package_path):
            if f.startswith('.'):
                continue
            try:
                proto_module = importlib.import_module(package_python_path + '.' + f.rstrip('.py'))
                for name, obj in inspect.getmembers(proto_module):
                    if inspect.isclass(obj) and name == buffer_name:
                        return obj
            except Exception as e:
                stat_logger.warning(e)
        else:
            return None

    def archive_model_base_path(self):
        return os.path.join(TEMP_DIRECTORY, "{}_{}".format(self.model_id, self.model_version))

    def archive_model_file_path(self):
        return "{}.{}".format(self.archive_model_base_path(), self.default_archive_format)
