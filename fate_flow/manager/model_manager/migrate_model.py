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
import base64
import shutil
from datetime import datetime

from fate_flow.utils.model_utils import gen_party_model_id

from arch.api.utils import dtable_utils
from federatedml.protobuf.generated import pipeline_pb2
from fate_flow.manager.model_manager import pipelined_model
from fate_arch.common.base_utils import json_loads, json_dumps
from arch.api.utils.file_utils import get_project_base_directory


def gen_model_file_path(model_id, model_version):
    return os.path.join(get_project_base_directory(), "model_local_cache", model_id, model_version)


def compare_roles(request_conf_roles: dict, run_time_conf_roles: dict):
    if request_conf_roles.keys() == run_time_conf_roles.keys():
        varify_format = True
        varify_equality = True
        for key in request_conf_roles.keys():
            varify_format = varify_format and (len(request_conf_roles[key]) == len(run_time_conf_roles[key])) and (isinstance(request_conf_roles[key], list))
            request_conf_roles_set = set(str(item) for item in request_conf_roles[key])
            run_time_conf_roles_set = set(str(item) for item in run_time_conf_roles[key])
            varify_equality = varify_equality and (request_conf_roles_set == run_time_conf_roles_set)
        if not varify_format:
            raise Exception("The structure of roles data of local configuration is different from "
                            "model runtime configuration's. Migration aborting.")
        else:
            return varify_equality
    raise Exception("The structure of roles data of local configuration is different from "
                    "model runtime configuration's. Migration aborting.")


def import_from_files(config: dict):
    model = pipelined_model.PipelinedModel(model_id=config["model_id"],
                                           model_version=config["model_version"])
    if config['force']:
        model.force = True
    model.unpack_model(config["file"])


def import_from_db(config: dict):
    model_path = gen_model_file_path(config["model_id"], config["model_version"])
    if config['force']:
        os.rename(model_path, model_path + '_backup_{}'.format(datetime.now().strftime('%Y%m%d%H%M')))


def migration(config_data: dict):
    try:
        party_model_id = gen_party_model_id(model_id=config_data["model_id"],
                                            role=config_data["local"]["role"],
                                            party_id=config_data["local"]["party_id"])
        model = pipelined_model.PipelinedModel(model_id=party_model_id,
                                               model_version=config_data["model_version"])
        if not model.exists():
            raise Exception("Can not found {} {} model local cache".format(config_data["model_id"],
                                                                           config_data["model_version"]))
        model_data = model.collect_models(in_bytes=True)
        if "pipeline.pipeline:Pipeline" not in model_data:
            raise Exception("Can not found pipeline file in model.")

        buffer_object_bytes = base64.b64decode(model_data["pipeline.pipeline:Pipeline"].encode())
        pipeline = pipeline_pb2.Pipeline()
        pipeline.ParseFromString(buffer_object_bytes)
        train_runtime_conf = json_loads(pipeline.train_runtime_conf)
    except Exception as e:
        return 100, str(e), {}
    else:
        # Generate 1. new job_id as new model version, 2. new model id (file path)
        previous_model_path = model.model_path

        model.model_id = dtable_utils.gen_party_namespace(config_data["migrate_role"], "model",
                                                          config_data["local"]["role"],
                                                          config_data["local"]["migrate_party_id"])
        model.model_version = config_data["unify_model_version"]

        # Copy the older version of files of models to the new dirpath
        model.set_model_path()
        shutil.copytree(src=previous_model_path, dst=model.model_path)

        # Utilize Pipeline_model collect model data. And modify related inner information of model
        train_runtime_conf["role"] = config_data["migrate_role"]
        train_runtime_conf["job_parameters"]["model_id"] = dtable_utils.all_party_key(train_runtime_conf["role"]) + "#model"
        train_runtime_conf["job_parameters"]["model_version"] = model.model_version

        pipeline.train_runtime_conf = json_dumps(train_runtime_conf, byte=True)
        pipeline.model_id = bytes(train_runtime_conf["job_parameters"]["model_id"], "utf-8")
        pipeline.model_version = bytes(train_runtime_conf["job_parameters"]["model_version"], "utf-8")
        model.save_pipeline(pipeline)
        shutil.copyfile(os.path.join(model.model_path, "pipeline.pb"),
                        os.path.join(model.model_path, "variables", "data", "pipeline", "pipeline", "Pipeline"))

        return 0, "Migrating model successfully. " \
                  "The configuration of model has been modified automatically. " \
                  "New model id is: {}, model version is: {}. " \
                  "Model files can be found at '{}'.".format(
            train_runtime_conf["job_parameters"]["model_id"],
            model.model_version, model.model_path), {"model_id": model.model_id,
                                                     "model_version": model.model_version,
                                                     "path": model.model_path}
