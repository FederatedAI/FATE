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
import json
import os
import tarfile
import tempfile
import yaml
from ..utils.uri_tools import parse_uri, replace_uri_path, get_schema_from_uri
from ..utils.file_utils import construct_local_dir
from ..conf.types import UriTypes
from ..entity.model_structure import MLModelSpec


class LocalFSModelManager(object):
    @classmethod
    def generate_output_model_uri(cls, output_dir_uri: str, job_id: str, task_name: str,
                                  role: str, party_id: str):
        model_id = "_".join([job_id, task_name, role, str(party_id)])
        model_version = "0"
        uri_obj = parse_uri(output_dir_uri)
        local_path = construct_local_dir(uri_obj.path, *[model_id, model_version])
        uri_obj = replace_uri_path(uri_obj, str(local_path))
        return uri_obj.geturl()

    @classmethod
    def get_output_model(cls, output_dir_uri):
        uri_obj = parse_uri(output_dir_uri)
        models = dict()
        with tempfile.TemporaryDirectory() as temp_dir:
            tar = tarfile.open(uri_obj.path, "r:")
            tar.extractall(path=temp_dir)
            tar.close()
            for file_name in os.listdir(temp_dir):
                if file_name.endswith("FMLModel.yaml"):
                    with open(os.path.join(temp_dir, file_name), "r") as fp:
                        model_meta = yaml.safe_load(fp)
                        model_spec = MLModelSpec.parse_obj(model_meta)

            for model in model_spec.party.models:
                file_format = model.file_format
                model_name = model.name

                if file_format == "json":
                    with open(os.path.join(temp_dir, model_name), "r") as fp:
                        models[model_name] = json.loads(fp.read())

        return models


def get_model_manager(model_uri: str):
    uri_type = get_schema_from_uri(model_uri)
    if uri_type == UriTypes.LOCAL:
        return LocalFSModelManager

