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
from fate_arch.common.base_utils import json_loads
from fate_arch.common.file_utils import get_project_base_directory
from fate_flow.pipelined_model.pipelined_model import PipelinedModel

from fate_flow.db.db_models import DB, MachineLearningModelInfo as MLModel
from fate_flow.utils import schedule_utils

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


def query_model_info_from_file(model_id, model_version, role=None, party_id=None, query_filters=None, check=False):
    if role and party_id:
        res = advanced_search_from_file(model_id=model_id, model_version=model_version, role=role,
                                        party_id=party_id, query_filters=query_filters, check=check)
    else:
        res = fuzzy_search_from_file(model_id=model_id, model_version=model_version,
                                     query_filters=query_filters, check=check)
    if res:
        return 0, 'Query model info from local model success.', res
    return 100, 'Query model info failed, cannot find model from local model files.', []


def advanced_search_from_file(model_id, model_version, role=None, party_id=None, query_filters=None, check=False):
    res = []
    party_model_id = gen_party_model_id(model_id=model_id, role=role, party_id=party_id)
    pipeline_model = PipelinedModel(model_id=party_model_id, model_version=model_version)
    model_info = gather_model_info_data(pipeline_model, query_filters=query_filters, check=check)
    if model_info:
        res.append(model_info)
    return res


def fuzzy_search_from_file(model_id, model_version, query_filters=None, check=False):
    res = {} if check else []
    model_dir = os.path.join(get_project_base_directory(), 'model_local_cache')
    model_fp_list = glob.glob(model_dir + f'/*#{model_id}/{model_version}')
    if model_fp_list:
        for fp in model_fp_list:
            pipeline_model = PipelinedModel(model_id=fp.split('/')[-2], model_version=fp.split('/')[-1])
            model_info = gather_model_info_data(pipeline_model, query_filters=query_filters, check=check)
            if model_info:
                if isinstance(res, dict):
                    res[fp] = model_info
                else:
                    res.append(model_info)
    return res


def gather_model_info_data(model: PipelinedModel, query_filters=None, check=False):
    if model.exists():
        pipeline = model.read_component_model('pipeline', 'pipeline')['Pipeline']
        if check:
            ver_list = pipeline.fate_version.split('.')
            if not (int(ver_list[1]) > 5 or (int(ver_list[1]) >= 5 and int(ver_list[2]) >= 1)):
                raise Exception(f"fate version ({pipeline.fate_version}) of model {pipeline.model_id} {pipeline.model_version} is older than 1.5.1")
        model_info = {}
        if query_filters and isinstance(query_filters, list):
            for attr, field in pipeline.ListFields():
                if attr.name in query_filters:
                    if isinstance(field, bytes):
                        model_info[attr.name] = json_loads(field)
                    else:
                        model_info[attr.name] = field
        else:
            for attr, field in pipeline.ListFields():
                if isinstance(field, bytes):
                    model_info[attr.name] = json_loads(field)
                else:
                    model_info[attr.name] = field
        return model_info
    return []


def query_model_info(model_version, role=None, party_id=None, model_id=None, query_filters=None):
    retcode, retmsg, data = query_model_info_from_db(role=role, party_id=party_id, model_id=model_id,
                                                     model_version=model_version, query_filters=query_filters)
    if not retcode:
        return retcode, retmsg, data
    else:
        aruments = locals()
        cond_attrs = [attr for attr in ['model_version', 'model_id'] if aruments[attr]]
        if 'model_version' in cond_attrs and 'model_id' in cond_attrs:
            retcode, retmsg, data = query_model_info_from_file(role=role, party_id=party_id, model_id=model_id,
                                                               model_version=model_version, query_filters=query_filters)
            if not retcode:
                return retcode, retmsg, data
            return retcode, 'Query model info failed, cannot find model neither from db nor local models', data
        else:
            return 100, 'Query model info failed, cannot find model from db. ' \
                        'Try use both model id and model version to query model info from local models', []


@DB.connection_context()
def sink_model_info_to_db():
    # TODO insert data of model info into model table
    pass


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


def get_predict_conf(model_id, model_version):
    model_dir = os.path.join(get_project_base_directory(), 'model_local_cache')
    model_fp_list = glob.glob(model_dir + f'/guest#*#{model_id}/{model_version}')
    if model_fp_list:
        fp = model_fp_list[0]
        pipeline_model = PipelinedModel(model_id=fp.split('/')[-2], model_version=fp.split('/')[-1])
        pipeline = pipeline_model.read_component_model('pipeline', 'pipeline')['Pipeline']
        predict_dsl = json_loads(pipeline.inference_dsl)

        train_runtime_conf = json_loads(pipeline.train_runtime_conf)
        parser = schedule_utils.get_dsl_parser_by_version(train_runtime_conf.get('dsl_version', '1') )
        return parser.generate_predict_conf_template(predict_dsl=predict_dsl, train_conf=train_runtime_conf,
                                                     model_id=model_id, model_version=model_version)
    else:
        return ''

