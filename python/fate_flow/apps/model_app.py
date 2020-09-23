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
import traceback

import peewee
from copy import deepcopy

from fate_arch.common.base_utils import json_loads
from fate_flow.db.db_models import MachineLearningModelInfo as MLModel
from fate_flow.db.db_models import Tag, DB, ModelTag, ModelOperationLog as OperLog
from flask import Flask, request, send_file

from fate_flow.pipelined_model.migrate_model import compare_roles
from fate_flow.scheduler import DAGScheduler
from fate_flow.settings import stat_logger, MODEL_STORE_ADDRESS, TEMP_DIRECTORY
from fate_flow.pipelined_model import migrate_model, pipelined_model, publish_model
from fate_flow.utils.api_utils import get_json_result, federated_api, error_response
from fate_flow.utils import job_utils
from fate_flow.utils.service_utils import ServiceUtils
from fate_flow.utils.detect_utils import check_config
from fate_flow.utils.model_utils import gen_party_model_id
from fate_flow.entity.types import ModelOperation, TagOperation
from fate_arch.common import file_utils, WorkMode, FederatedMode

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/load', methods=['POST'])
def load_model():
    request_config = request.json
    if request_config.get('job_id', None):
        with DB.connection_context():
            model = MLModel.get_or_none(
                MLModel.f_job_id == request_config.get("job_id"),
                MLModel.f_role == 'guest'
            )
        if model:
            model_info = model.to_json()
            request_config['initiator'] = {}
            request_config['initiator']['party_id'] = str(model_info.get('f_initiator_party_id'))
            request_config['initiator']['role'] = model_info.get('f_initiator_role')
            request_config['job_parameters'] = model_info.get('f_runtime_conf').get('job_parameters')
            request_config['role'] = model_info.get('f_runtime_conf').get('role')
            for key, value in request_config['role'].items():
                for i, v in enumerate(value):
                    value[i] = str(v)
            request_config.pop('job_id')
        else:
            return get_json_result(retcode=101,
                                   retmsg="model with version {} can not be found in database. "
                                          "Please check if the model version is valid.".format(request_config.get('job_id')))
    _job_id = job_utils.generate_job_id()
    initiator_party_id = request_config['initiator']['party_id']
    initiator_role = request_config['initiator']['role']
    publish_model.generate_publish_model_info(request_config)
    load_status = True
    load_status_info = {}
    load_status_msg = 'success'
    load_status_info['detail'] = {}
    if "federated_mode" not in request_config['job_parameters']:
        if request_config["job_parameters"]["work_mode"] == WorkMode.STANDALONE:
            request_config['job_parameters']["federated_mode"] = FederatedMode.SINGLE
        elif request_config["job_parameters"]["work_mode"] == WorkMode.CLUSTER:
            request_config['job_parameters']["federated_mode"] = FederatedMode.MULTIPLE
    for role_name, role_partys in request_config.get("role").items():
        if role_name == 'arbiter':
            continue
        load_status_info[role_name] = load_status_info.get(role_name, {})
        load_status_info['detail'][role_name] = {}
        for _party_id in role_partys:
            request_config['local'] = {'role': role_name, 'party_id': _party_id}
            try:
                response = federated_api(job_id=_job_id,
                                         method='POST',
                                         endpoint='/model/load/do',
                                         src_party_id=initiator_party_id,
                                         dest_party_id=_party_id,
                                         src_role = initiator_role,
                                         json_body=request_config,
                                         federated_mode=request_config['job_parameters']['federated_mode'])
                load_status_info[role_name][_party_id] = response['retcode']
                detail = {_party_id: {}}
                detail[_party_id]['retcode'] = response['retcode']
                detail[_party_id]['retmsg'] = response['retmsg']
                load_status_info['detail'][role_name].update(detail)
                if response['retcode']:
                    load_status = False
                    load_status_msg = 'failed'
            except Exception as e:
                stat_logger.exception(e)
                load_status = False
                load_status_msg = 'failed'
                load_status_info[role_name][_party_id] = 100
    return get_json_result(job_id=_job_id, retcode=(0 if load_status else 101), retmsg=load_status_msg,
                           data=load_status_info)


@manager.route('/migrate', methods=['POST'])
def migrate_model_process():
    request_config = request.json
    _job_id = job_utils.generate_job_id()
    initiator_party_id = request_config['migrate_initiator']['party_id']
    initiator_role = request_config['migrate_initiator']['role']
    if not request_config.get("unify_model_version"):
        request_config["unify_model_version"] = _job_id
    migrate_status = True
    migrate_status_info = {}
    migrate_status_msg = 'success'
    migrate_status_info['detail'] = {}

    require_arguments = ["migrate_initiator", "role", "migrate_role", "model_id",
                         "model_version", "execute_party", "job_parameters"]
    check_config(request_config, require_arguments)

    try:
        if compare_roles(request_config.get("migrate_role"), request_config.get("role")):
            return get_json_result(retcode=100,
                                   retmsg="The config of previous roles is the same with that of migrate roles. "
                                          "There is no need to migrate model. Migration process aborting.")
    except Exception as e:
        return get_json_result(retcode=100, retmsg=str(e))

    local_template = {
        "role": "",
        "party_id": "",
        "migrate_party_id": ""
    }

    res_dict = {}

    for role_name, role_partys in request_config.get("migrate_role").items():
        for offset, party_id in enumerate(role_partys):
            local_res = deepcopy(local_template)
            local_res["role"] = role_name
            local_res["party_id"] = request_config.get("role").get(role_name)[offset]
            local_res["migrate_party_id"] = party_id
            if not res_dict.get(role_name):
                res_dict[role_name] = {}
            res_dict[role_name][local_res["party_id"]] = local_res

    for role_name, role_partys in request_config.get("execute_party").items():
        migrate_status_info[role_name] = migrate_status_info.get(role_name, {})
        migrate_status_info['detail'][role_name] = {}
        for party_id in role_partys:
            request_config["local"] = res_dict.get(role_name).get(party_id)
            try:
                response = federated_api(job_id=_job_id,
                                         method='POST',
                                         endpoint='/model/migrate/do',
                                         src_party_id=initiator_party_id,
                                         dest_party_id=party_id,
                                         src_role=initiator_role,
                                         json_body=request_config,
                                         federated_mode=request_config['job_parameters']['federated_mode'])
                migrate_status_info[role_name][party_id] = response['retcode']
                detail = {party_id: {}}
                detail[party_id]['retcode'] = response['retcode']
                detail[party_id]['retmsg'] = response['retmsg']
                migrate_status_info['detail'][role_name].update(detail)
            except Exception as e:
                stat_logger.exception(e)
                migrate_status = False
                migrate_status_msg = 'failed'
                migrate_status_info[role_name][party_id] = 100
    return get_json_result(job_id=_job_id, retcode=(0 if migrate_status else 101),
                           retmsg=migrate_status_msg, data=migrate_status_info)


@manager.route('/migrate/do', methods=['POST'])
def do_migrate_model():
    request_data = request.json
    retcode, retmsg, data = migrate_model.migration(config_data=request_data)
    operation_record(request_data, "migrate", "success" if not retcode else "failed")
    return get_json_result(retcode=retcode, retmsg=retmsg, data=data)


@manager.route('/load/do', methods=['POST'])
def do_load_model():
    request_data = request.json
    request_data["servings"] = ServiceUtils.get("servings", {}).get('hosts', [])
    retcode, retmsg = publish_model.load_model(config_data=request_data)
    try:
        if not retcode:
            with DB.connection_context():
                model = MLModel.get_or_none(MLModel.f_role == request_data.get("local").get("role"),
                                            MLModel.f_party_id == request_data.get("local").get("party_id"),
                                            MLModel.f_model_id == request_data.get("job_parameters").get("model_id"),
                                            MLModel.f_model_version == request_data.get("job_parameters").get("model_version"))
                if model:
                    count = model.f_loaded_times
                    model.f_loaded_times = count + 1
                    model.save()
    except Exception as modify_err:
        stat_logger.exception(modify_err)

    try:
        party_model_id = gen_party_model_id(role=request_data.get("local").get("role"),
                                            party_id=request_data.get("local").get("party_id"),
                                            model_id=request_data.get("job_parameters").get("model_id"))
        src_model_path = os.path.join(file_utils.get_project_base_directory(), 'model_local_cache', party_model_id,
                                      request_data.get("job_parameters").get("model_version"))
        dst_model_path = os.path.join(file_utils.get_project_base_directory(), 'loaded_model_backup',
                                      party_model_id, request_data.get("job_parameters").get("model_version"))
        if not os.path.exists(dst_model_path):
            shutil.copytree(src=src_model_path, dst=dst_model_path)
    except Exception as copy_err:
        stat_logger.exception(copy_err)
    operation_record(request_data, "load", "success" if not retcode else "failed")
    return get_json_result(retcode=retcode, retmsg=retmsg)


@manager.route('/bind', methods=['POST'])
def bind_model_service():
    request_config = request.json
    if request_config.get('job_id', None):
        with DB.connection_context():
            model = MLModel.get_or_none(
                MLModel.f_job_id == request_config.get("job_id"),
                MLModel.f_role == 'guest'
            )
        if model:
            model_info = model.to_json()
            request_config['initiator'] = {}
            request_config['initiator']['party_id'] = str(model_info.get('f_initiator_party_id'))
            request_config['initiator']['role'] = model_info.get('f_initiator_role')
            request_config['job_parameters'] = model_info.get('f_runtime_conf').get('job_parameters')
            request_config['role'] = model_info.get('f_runtime_conf').get('role')
            for key, value in request_config['role'].items():
                for i, v in enumerate(value):
                    value[i] = str(v)
            request_config.pop('job_id')
        else:
            return get_json_result(retcode=101,
                                   retmsg="model {} can not be found in database. "
                                          "Please check if the model version is valid.".format(request_config.get('job_id')))
    if not request_config.get('servings'):
        # get my party all servings
        request_config['servings'] = ServiceUtils.get("servings", {}).get('hosts', [])
    service_id = request_config.get('service_id')
    if not service_id:
        return get_json_result(retcode=101, retmsg='no service id')
    check_config(request_config, ['initiator', 'role', 'job_parameters'])
    bind_status, retmsg = publish_model.bind_model_service(config_data=request_config)
    operation_record(request_config, "bind", "success" if not bind_status else "failed")
    return get_json_result(retcode=bind_status, retmsg='service id is {}'.format(service_id) if not retmsg else retmsg)


@manager.route('/transfer', methods=['post'])
def transfer_model():
    model_data = publish_model.download_model(request.json)
    return get_json_result(retcode=0, retmsg="success", data=model_data)


@manager.route('/<model_operation>', methods=['post', 'get'])
def operate_model(model_operation):
    request_config = request.json or request.form.to_dict()
    job_id = job_utils.generate_job_id()
    if model_operation not in [ModelOperation.STORE, ModelOperation.RESTORE, ModelOperation.EXPORT, ModelOperation.IMPORT]:
        raise Exception('Can not support this operating now: {}'.format(model_operation))
    required_arguments = ["model_id", "model_version", "role", "party_id"]
    check_config(request_config, required_arguments=required_arguments)
    request_config["model_id"] = gen_party_model_id(model_id=request_config["model_id"], role=request_config["role"], party_id=request_config["party_id"])
    if model_operation in [ModelOperation.EXPORT, ModelOperation.IMPORT]:
        if model_operation == ModelOperation.IMPORT:
            try:
                file = request.files.get('file')
                file_path = os.path.join(TEMP_DIRECTORY, file.filename)
                # if not os.path.exists(file_path):
                #     raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                #                     'please check the path: {}'.format(file_path))
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    file.save(file_path)
                except Exception as e:
                    shutil.rmtree(file_path)
                    raise e
                request_config['file'] = file_path
                model = pipelined_model.PipelinedModel(model_id=request_config["model_id"], model_version=request_config["model_version"])
                model.unpack_model(file_path)

                pipeline = model.read_component_model('pipeline', 'pipeline')['Pipeline']
                train_runtime_conf = json_loads(pipeline.train_runtime_conf)
                permitted_party_id = []
                for key, value in train_runtime_conf.get('role', {}).items():
                    for v in value:
                        permitted_party_id.extend([v, str(v)])
                if request_config["party_id"] not in permitted_party_id:
                    shutil.rmtree(model.model_path)
                    raise Exception("party id {} is not in model roles, please check if the party id is valid.")
                try:
                    with DB.connection_context():
                        model = MLModel.get_or_none(
                            MLModel.f_job_id == train_runtime_conf["job_parameters"]["model_version"],
                            MLModel.f_role == request_config["role"]
                        )
                        if not model:
                            MLModel.create(
                                f_role=request_config["role"],
                                f_party_id=request_config["party_id"],
                                f_roles=train_runtime_conf["role"],
                                f_job_id=train_runtime_conf["job_parameters"]["model_version"],
                                f_model_id=train_runtime_conf["job_parameters"]["model_id"],
                                f_model_version=train_runtime_conf["job_parameters"]["model_version"],
                                f_initiator_role=train_runtime_conf["initiator"]["role"],
                                f_initiator_party_id=train_runtime_conf["initiator"]["party_id"],
                                f_runtime_conf=train_runtime_conf,
                                f_work_mode=train_runtime_conf["job_parameters"]["work_mode"],
                                f_dsl=json_loads(pipeline.train_dsl),
                                f_imported=1,
                                f_job_status='complete'
                            )
                        else:
                            stat_logger.info(f'job id: {train_runtime_conf["job_parameters"]["model_version"]}, '
                                             f'role: {request_config["role"]} model info already existed in database.')
                except peewee.IntegrityError as e:
                    stat_logger.exception(e)
                operation_record(request_config, "import", "success")
                return get_json_result()
            except Exception:
                operation_record(request_config, "import", "failed")
                raise
        else:
            try:
                model = pipelined_model.PipelinedModel(model_id=request_config["model_id"], model_version=request_config["model_version"])
                if model.exists():
                    archive_file_path = model.packaging_model()
                    operation_record(request_config, "export", "success")
                    return send_file(archive_file_path, attachment_filename=os.path.basename(archive_file_path), as_attachment=True)
                else:
                    operation_record(request_config, "export", "failed")
                    res = error_response(response_code=210,
                                         retmsg="Model {} {} is not exist.".format(request_config.get("model_id"),
                                                                                    request_config.get("model_version")))
                    return res
            except Exception as e:
                operation_record(request_config, "export", "failed")
                stat_logger.exception(e)
                return error_response(response_code=210, retmsg=str(e))
    else:
        data = {}
        job_dsl, job_runtime_conf = gen_model_operation_job_config(request_config, model_operation)
        job_id, job_dsl_path, job_runtime_conf_path, logs_directory, model_info, board_url = DAGScheduler.submit(
            {'job_dsl': job_dsl, 'job_runtime_conf': job_runtime_conf}, job_id=job_id)
        data.update({'job_dsl_path': job_dsl_path, 'job_runtime_conf_path': job_runtime_conf_path,
                     'board_url': board_url, 'logs_directory': logs_directory})
        operation_record(data=job_runtime_conf, oper_type=model_operation, oper_status='')
        return get_json_result(job_id=job_id, data=data)


@manager.route('/model_tag/<operation>', methods=['POST'])
@DB.connection_context()
def tag_model(operation):
    if operation not in ['retrieve', 'create', 'remove']:
        return get_json_result(100, "'{}' is not currently supported.".format(operation))

    request_data = request.json
    model = MLModel.get_or_none(MLModel.f_job_id == request_data.get("job_id"))
    if not model:
        raise Exception("Can not found model by job id: '{}'.".format(request_data.get("job_id")))

    if operation == 'retrieve':
        res = {'tags': []}
        tags = (Tag.select().join(ModelTag, on=ModelTag.f_t_id == Tag.f_id).where(ModelTag.f_m_id == model.f_id))
        for tag in tags:
            res['tags'].append({'name': tag.f_name, 'description': tag.f_desc})
        res['count'] = tags.count()
        return get_json_result(data=res)
    elif operation == 'remove':
        tag = Tag.get_or_none(Tag.f_name == request_data.get('tag_name'))
        if not tag:
            raise Exception("Can not found '{}' tag.".format(request_data.get('tag_name')))
        tags = (Tag.select().join(ModelTag, on=ModelTag.f_t_id == Tag.f_id).where(ModelTag.f_m_id == model.f_id))
        if tag.f_name not in [t.f_name for t in tags]:
            raise Exception("Model {} {} does not have tag '{}'.".format(model.f_model_id,
                                                                         model.f_model_version,
                                                                         tag.f_name))
        delete_query = ModelTag.delete().where(ModelTag.f_m_id == model.f_id, ModelTag.f_t_id == tag.f_id)
        delete_query.execute()
        return get_json_result(retmsg="'{}' tag has been removed from tag list of model {} {}.".format(request_data.get('tag_name'),
                                                                                                       model.f_model_id,
                                                                                                       model.f_model_version))
    else:
        if not str(request_data.get('tag_name')):
            raise Exception("Tag name should not be an empty string.")
        tag = Tag.get_or_none(Tag.f_name == request_data.get('tag_name'))
        if not tag:
            tag = Tag()
            tag.f_name = request_data.get('tag_name')
            tag.save(force_insert=True)
        else:
            tags = (Tag.select().join(ModelTag, on=ModelTag.f_t_id == Tag.f_id).where(ModelTag.f_m_id == model.f_id))
            if tag.f_name in [t.f_name for t in tags]:
                raise Exception("Model {} {} already been tagged as tag '{}'.".format(model.f_model_id,
                                                                                      model.f_model_version,
                                                                                      tag.f_name))
        ModelTag.create(f_t_id=tag.f_id, f_m_id=model.f_id)
        return get_json_result(retmsg="Adding {} tag for model with job id: {} successfully.".format(request_data.get('tag_name'),
                                                                                                     request_data.get('job_id')))


@manager.route('/tag/<tag_operation>', methods=['POST'])
@DB.connection_context()
def operate_tag(tag_operation):
    request_data = request.json
    if tag_operation not in [TagOperation.CREATE, TagOperation.RETRIEVE, TagOperation.UPDATE,
                             TagOperation.DESTROY, TagOperation.LIST]:
        raise Exception('The {} operation is not currently supported.'.format(tag_operation))

    tag_name = request_data.get('tag_name')
    tag_desc = request_data.get('tag_desc')
    if tag_operation == TagOperation.CREATE:
        try:
            if not tag_name:
                return get_json_result(100, "'{}' tag created failed. Please input a valid tag name.".format(tag_name))
            else:
                Tag.create(f_name=tag_name, f_desc=tag_desc)
        except peewee.IntegrityError:
            raise Exception("'{}' has already exists in database.".format(tag_name))
        else:
            return get_json_result("'{}' tag has been created successfully.".format(tag_name))

    elif tag_operation == TagOperation.LIST:
        tags = Tag.select()
        limit = request_data.get('limit')
        res = {"tags": []}

        if limit > len(tags):
            count = len(tags)
        else:
            count = limit
        for tag in tags[:count]:
            res['tags'].append({'name': tag.f_name, 'description': tag.f_desc,
                                'model_count': ModelTag.filter(ModelTag.f_t_id == tag.f_id).count()})
        return get_json_result(data=res)

    else:
        if not (tag_operation == TagOperation.RETRIEVE and not request_data.get('with_model')):
            try:
                tag = Tag.get(Tag.f_name == tag_name)
            except peewee.DoesNotExist:
                raise Exception("Can not found '{}' tag.".format(tag_name))

        if tag_operation == TagOperation.RETRIEVE:
            if request_data.get('with_model', False):
                res = {'models': []}
                models = (MLModel.select().join(ModelTag, on=ModelTag.f_m_id == MLModel.f_id).where(ModelTag.f_t_id == tag.f_id))
                for model in models:
                        res["models"].append({
                        "model_id": model.f_model_id,
                        "model_version": model.f_model_version,
                        "model_size": model.f_size
                    })
                res["count"] = models.count()
                return get_json_result(data=res)
            else:
                tags = Tag.filter(Tag.f_name.contains(tag_name))
                if not tags:
                    return get_json_result(100, retmsg="No tags found.")
                res = {'tags': []}
                for tag in tags:
                    res['tags'].append({'name': tag.f_name, 'description': tag.f_desc})
                return get_json_result(data=res)

        elif tag_operation == TagOperation.UPDATE:
            new_tag_name = request_data.get('new_tag_name', None)
            new_tag_desc = request_data.get('new_tag_desc', None)
            if (tag.f_name == new_tag_name) and (tag.f_desc == new_tag_desc):
                return get_json_result(100, "Nothing to be updated.")
            else:
                if request_data.get('new_tag_name'):
                    if not Tag.get_or_none(Tag.f_name == new_tag_name):
                        tag.f_name = new_tag_name
                    else:
                        return get_json_result(100, retmsg="'{}' tag already exists.".format(new_tag_name))

                tag.f_desc = new_tag_desc
                tag.save()
                return get_json_result(retmsg="Infomation of '{}' tag has been updated successfully.".format(tag_name))

        else:
            delete_query = ModelTag.delete().where(ModelTag.f_t_id == tag.f_id)
            delete_query.execute()
            Tag.delete_instance(tag)
            return get_json_result(retmsg="'{}' tag has been deleted successfully.".format(tag_name))


def gen_model_operation_job_config(config_data: dict, model_operation: ModelOperation):
    job_runtime_conf = job_utils.runtime_conf_basic(if_local=True)
    initiator_role = "local"
    job_dsl = {
        "components": {}
    }

    if model_operation in [ModelOperation.STORE, ModelOperation.RESTORE]:
        component_name = "{}_0".format(model_operation)
        component_parameters = dict()
        component_parameters["model_id"] = [config_data["model_id"]]
        component_parameters["model_version"] = [config_data["model_version"]]
        component_parameters["store_address"] = [MODEL_STORE_ADDRESS]
        if model_operation == ModelOperation.STORE:
            component_parameters["force_update"] = [config_data.get("force_update", False)]
        job_runtime_conf["role_parameters"][initiator_role] = {component_name: component_parameters}
        job_dsl["components"][component_name] = {
            "module": "Model{}".format(model_operation.capitalize())
        }
    else:
        raise Exception("Can not support this model operation: {}".format(model_operation))
    return job_dsl, job_runtime_conf


@DB.connection_context()
def operation_record(data: dict, oper_type, oper_status):
    try:
        if oper_type == 'migrate':
            OperLog.create(f_operation_type=oper_type,
                           f_operation_status=oper_status,
                           f_initiator_role=data.get("migrate_initiator", {}).get("role"),
                           f_initiator_party_id=data.get("migrate_initiator", {}).get("party_id"),
                           f_request_ip=request.remote_addr,
                           f_model_id=data.get("model_id"),
                           f_model_version=data.get("model_version"))
        elif oper_type == 'load':
            OperLog.create(f_operation_type=oper_type,
                           f_operation_status=oper_status,
                           f_initiator_role=data.get("initiator").get("role"),
                           f_initiator_party_id=data.get("initiator").get("party_id"),
                           f_request_ip=request.remote_addr,
                           f_model_id=data.get("job_parameters").get("model_id"),
                           f_model_version=data.get("job_parameters").get("model_version"))
        else:
            OperLog.create(f_operation_type=oper_type,
                           f_operation_status=oper_status,
                           f_initiator_role=data.get("role") if data.get("role") else data.get("initiator").get("role"),
                           f_initiator_party_id=data.get("party_id") if data.get("party_id") else data.get("initiator").get("party_id"),
                           f_request_ip=request.remote_addr,
                           f_model_id=data.get("model_id") if data.get("model_id") else data.get("job_parameters").get("model_id"),
                           f_model_version=data.get("model_version") if data.get("model_version") else data.get("job_parameters").get("model_version"))
    except Exception:
        stat_logger.error(traceback.format_exc())
