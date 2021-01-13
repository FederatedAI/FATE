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
import glob
import operator
from collections import OrderedDict

import peewee
from fate_arch.common.log import sql_logger
from fate_flow.settings import stat_logger
from fate_arch.common.base_utils import json_loads, current_timestamp
from fate_arch.common.file_utils import get_project_base_directory
from fate_flow.pipelined_model.pipelined_model import PipelinedModel

from fate_flow.db.db_models import DB, MachineLearningModelInfo as MLModel

gen_key_string_separator = '#'


def gen_party_model_id(model_id, role, party_id):
    return gen_key_string_separator.join([role, str(party_id), model_id]) if model_id else None


def gen_model_id(all_party):
    return gen_key_string_separator.join([all_party_key(all_party), "model"])


def all_party_key(all_party):
    """
    Join all party as party key
    :param all_party:
        "role": {
            "guest": [9999],
            "host": [10000],
            "arbiter": [10000]
         }
    :return:
    """
    if not all_party:
        all_party_key = 'all'
    elif isinstance(all_party, dict):
        sorted_role_name = sorted(all_party.keys())
        all_party_key = gen_key_string_separator.join([
            ('%s-%s' % (
                role_name,
                '_'.join([str(p) for p in sorted(set(all_party[role_name]))]))
             )
            for role_name in sorted_role_name])
    else:
        all_party_key = None
    return all_party_key


@DB.connection_context()
def query_model_info_from_db(model_version, role=None, party_id=None, model_id=None, query_filters=None, **kwargs):
    conditions = []
    filters = []
    aruments = locals()
    cond_attrs = [attr for attr in ['model_version', 'model_id', 'role', 'party_id'] if aruments[attr]]
    for f_n in cond_attrs:
        conditions.append(operator.attrgetter('f_%s' % f_n)(MLModel) == aruments[f_n])
    for f_n in kwargs:
        if hasattr(MLModel, 'f_%s' % f_n):
            conditions.append(operator.attrgetter('f_%s' % f_n)(MLModel))

    if query_filters and isinstance(query_filters, list):
        for attr in query_filters:
            attr_name = 'f_%s' % attr
            if hasattr(MLModel, attr_name):
                filters.append(operator.attrgetter(attr_name)(MLModel))

    if filters:
        models = MLModel.select(*filters).where(*conditions)
    else:
        models = MLModel.select().where(*conditions)

    if models:
        return 0, 'Query model info from db success.', [model.to_json() for model in models]
    else:
        return 100, 'Query model info failed, cannot find model from db. ', []


def query_model_info_from_file(model_id=None, model_version=None, role=None, party_id=None, query_filters=None, to_dict=False, **kwargs):
    res = {} if to_dict else []
    model_dir = os.path.join(get_project_base_directory(), 'model_local_cache')
    glob_dir = f"{model_dir}{os.sep}{role if role else '*'}#{party_id if party_id else '*'}#{model_id if model_id else '*'}{os.sep}{model_version if model_version else '*'}"
    stat_logger.info(f'glob model dir: {glob_dir}')
    model_fp_list = glob.glob(glob_dir)
    if model_fp_list:
        for fp in model_fp_list:
            pipeline_model = PipelinedModel(model_id=fp.split('/')[-2], model_version=fp.split('/')[-1])
            model_info = gather_model_info_data(pipeline_model, query_filters=query_filters)
            if model_info:
                if isinstance(res, dict):
                    res[fp] = model_info
                else:
                    res.append(model_info)

                if kwargs.get('save'):
                    try:
                        insert_info = gather_model_info_data(pipeline_model).copy()
                        insert_info['role'] = fp.split('/')[-2].split('#')[0]
                        insert_info['party_id'] = fp.split('/')[-2].split('#')[1]
                        insert_info['job_id'] = insert_info.get('f_model_version')
                        insert_info['size'] = pipeline_model.calculate_model_file_size()
                        if compare_version(insert_info['f_fate_version'], '1.5.1') == 'lt':
                            insert_info['roles'] = insert_info.get('f_train_runtime_conf', {}).get('role', {})
                            insert_info['initiator_role'] = insert_info.get('f_train_runtime_conf', {}).get('initiator', {}).get('role')
                            insert_info['initiator_party_id'] = insert_info.get('f_train_runtime_conf', {}).get('initiator', {}).get('party_id')
                        save_model_info(insert_info)
                    except Exception as e:
                        stat_logger.exception(e)
    if res:
        return 0, 'Query model info from local model success.', res
    return 100, 'Query model info failed, cannot find model from local model files.', res


def gather_model_info_data(model: PipelinedModel, query_filters=None):
    if model.exists():
        pipeline = model.read_component_model('pipeline', 'pipeline')['Pipeline']
        model_info = OrderedDict()
        if query_filters and isinstance(query_filters, list):
            for attr, field in pipeline.ListFields():
                if attr.name in query_filters:
                    if isinstance(field, bytes):
                        model_info["f_" + attr.name] = json_loads(field, OrderedDict)
                    else:
                        model_info["f_" + attr.name] = field
        else:
            for attr, field in pipeline.ListFields():
                if isinstance(field, bytes):
                    model_info["f_" + attr.name] = json_loads(field, OrderedDict)
                else:
                    model_info["f_" + attr.name] = field
        return model_info
    return []


def query_model_info(model_version, role=None, party_id=None, model_id=None, query_filters=None, **kwargs):
    arguments = locals()
    retcode, retmsg, data = query_model_info_from_db(**arguments)
    if not retcode:
        return retcode, retmsg, data
    else:
        arguments['save'] = True
        retcode, retmsg, data = query_model_info_from_file(**arguments)
        if not retcode:
            return retcode, retmsg, data
        return 100, 'Query model info failed, cannot find model from db. ' \
                    'Try use both model id and model version to query model info from local models', []


@DB.connection_context()
def save_model_info(model_info):
    model = MLModel()
    model.f_create_time = current_timestamp()
    for k, v in model_info.items():
        attr_name = 'f_%s' % k
        if hasattr(MLModel, attr_name):
            setattr(model, attr_name, v)
        elif hasattr(MLModel, k):
            setattr(model, k, v)
    try:
        rows = model.save(force_insert=True)
        if rows != 1:
            raise Exception("Create {} failed".format(MLModel))
        return model
    except peewee.IntegrityError as e:
        if e.args[0] == 1062:
            sql_logger(job_id=model_info.get("job_id", "fate_flow")).warning(e)
        else:
            raise Exception("Create {} failed:\n{}".format(MLModel, e))
    except Exception as e:
        raise Exception("Create {} failed:\n{}".format(MLModel, e))


def compare_version(version: str, target_version: str):
    ver_list = version.split('.')
    tar_ver_list = target_version.split('.')
    if int(ver_list[0]) >= int(tar_ver_list[0]):
        if int(ver_list[1]) > int(tar_ver_list[1]):
            return 'gt'
        elif int(ver_list[1]) < int(tar_ver_list[1]):
            return 'lt'
        else:
            if int(ver_list[2]) > int(tar_ver_list[2]):
                return 'gt'
            elif int(ver_list[2]) == int(tar_ver_list[2]):
                return 'eq'
            else:
                return 'lt'
    return 'lt'


def check_if_parent_model(pipeline):
    if compare_version(pipeline.fate_version, '1.5.0') == 'gt':
        if pipeline.parent:
            return True
    return False


def check_before_deploy(pipeline_model: PipelinedModel):
    pipeline = pipeline_model.read_component_model('pipeline', 'pipeline')['Pipeline']

    if compare_version(pipeline.fate_version, '1.5.0') == 'gt':
        if pipeline.parent:
            return True
    elif compare_version(pipeline.fate_version, '1.5.0') == 'eq':
        return True
    return False


def check_if_deployed(role, party_id, model_id, model_version):
    party_model_id = gen_party_model_id(model_id=model_id, role=role, party_id=party_id)
    pipeline_model = PipelinedModel(model_id=party_model_id, model_version=model_version)
    if not pipeline_model.exists():
        raise Exception(f"Model {party_model_id} {model_version} not exists in model local cache.")
    else:
        pipeline = pipeline_model.read_component_model('pipeline', 'pipeline')['Pipeline']
        return not check_if_parent_model(pipeline)