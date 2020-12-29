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
import shutil
from fate_arch.common import file_utils
from fate_flow.utils import model_utils
from fate_flow.settings import stat_logger
from fate_arch.common.base_utils import json_loads, json_dumps
from fate_flow.utils.config_adapter import JobRuntimeConfigAdapter
from fate_flow.pipelined_model.pipelined_model import PipelinedModel
from fate_flow.utils.model_utils import check_before_deploy
from fate_flow.utils.schedule_utils import get_dsl_parser_by_version


def deploy(config_data):
    model_id = config_data.get('model_id')
    model_version = config_data.get('model_version')
    local_role = config_data.get('local').get('role')
    local_party_id = config_data.get('local').get('party_id')
    child_model_version = config_data.get('child_model_version')

    try:
        party_model_id = model_utils.gen_party_model_id(model_id=model_id, role=local_role, party_id=local_party_id)
        model = PipelinedModel(model_id=party_model_id, model_version=model_version)
        model_data = model.collect_models(in_bytes=True)
        if "pipeline.pipeline:Pipeline" not in model_data:
            raise Exception("Can not found pipeline file in model.")

        # check if the model could be executed the deploy process (parent/child)
        if not check_before_deploy(model):
            raise Exception('Child model could not be deployed.')

        # copy proto content from parent model and generate a child model
        deploy_model = PipelinedModel(model_id=party_model_id, model_version=child_model_version)
        shutil.copytree(src=model.model_path, dst=deploy_model.model_path)
        pipeline = deploy_model.read_component_model('pipeline', 'pipeline')['Pipeline']

        # modify two pipeline files (model version/ train_runtime_conf)
        train_runtime_conf = json_loads(pipeline.train_runtime_conf)
        adapter = JobRuntimeConfigAdapter(train_runtime_conf)
        train_runtime_conf = adapter.update_model_id_version(model_version=deploy_model.model_version)
        pipeline.model_version = child_model_version
        pipeline.train_runtime_conf = json_dumps(train_runtime_conf, byte=True)

        #  save predict dsl into child model file
        parser = get_dsl_parser_by_version(train_runtime_conf.get('dsl_version', '1'))
        parser.verify_dsl(config_data.get('predict_dsl'), "predict")
        inference_dsl = parser.get_predict_dsl(role=local_role,
                                               predict_dsl=config_data.get('predict_dsl'),
                                               setting_conf_prefix=os.path.join(file_utils.get_python_base_directory(),
                                                                                *['federatedml', 'conf', 'setting_conf']))
        pipeline.inference_dsl = json_dumps(inference_dsl, byte=True)
        if model_utils.compare_version(pipeline.fate_version, '1.5.0') == 'gt':
            pipeline.parent_info = json_dumps({'parent_model_id': model_id, 'parent_model_version': model_version}, byte=True)
            pipeline.parent = False
            runtime_conf_on_party = json_loads(pipeline.runtime_conf_on_party)
            runtime_conf_on_party['job_parameters']['model_version'] = child_model_version
            pipeline.runtime_conf_on_party = json_dumps(runtime_conf_on_party, byte=True)

        # save model file
        deploy_model.save_pipeline(pipeline)
        shutil.copyfile(os.path.join(deploy_model.model_path, "pipeline.pb"),
                        os.path.join(deploy_model.model_path, "variables", "data", "pipeline", "pipeline", "Pipeline"))

        model_info = model_utils.gather_model_info_data(deploy_model)
        model_info['job_id'] = model_info['f_model_version']
        model_info['size'] = deploy_model.calculate_model_file_size()
        model_info['role'] = local_role
        model_info['party_id'] = local_party_id
        model_info['work_mode'] = adapter.get_job_work_mode()
        model_info['parent'] = False if model_info.get('f_inference_dsl') else True
        if model_utils.compare_version(model_info['f_fate_version'], '1.5.0') == 'eq':
            model_info['roles'] = model_info.get('f_train_runtime_conf', {}).get('role', {})
            model_info['initiator_role'] = model_info.get('f_train_runtime_conf', {}).get('initiator', {}).get('role')
            model_info['initiator_party_id'] = model_info.get('f_train_runtime_conf', {}).get('initiator', {}).get('party_id')
        model_utils.save_model_info(model_info)

    except Exception as e:
        stat_logger.exception(e)
        return 100, f"deploy model of role {local_role} {local_party_id} failed, details: {str(e)}"
    else:
        return 0, f"deploy model of role {local_role} {local_party_id} success"
