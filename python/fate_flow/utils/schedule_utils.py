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

from fate_arch.common import file_utils
from fate_flow.db.db_models import DB, Job
from fate_flow.scheduler.dsl_parser import DSLParser, DSLParserV2
from fate_flow.utils.config_adapter import JobRuntimeConfigAdapter


@DB.connection_context()
def get_job_dsl_parser_by_job_id(job_id):
    jobs = Job.select(Job.f_dsl, Job.f_runtime_conf_on_party, Job.f_train_runtime_conf).where(Job.f_job_id == job_id)
    if jobs:
        job = jobs[0]
        job_dsl_parser = get_job_dsl_parser(dsl=job.f_dsl, runtime_conf=job.f_runtime_conf_on_party,
                                            train_runtime_conf=job.f_train_runtime_conf)
        return job_dsl_parser
    else:
        return None


def get_job_dsl_parser(dsl=None, runtime_conf=None, pipeline_dsl=None, train_runtime_conf=None):
    parser_version = str(runtime_conf.get('dsl_version', '1'))
    dsl_parser = get_dsl_parser_by_version(parser_version)
    default_runtime_conf_path = os.path.join(file_utils.get_python_base_directory(),
                                             *['federatedml', 'conf', 'default_runtime_conf'])
    job_type = JobRuntimeConfigAdapter(runtime_conf).get_job_type()
    dsl_parser.run(dsl=dsl,
                   runtime_conf=runtime_conf,
                   pipeline_dsl=pipeline_dsl,
                   pipeline_runtime_conf=train_runtime_conf,
                   default_runtime_conf_prefix=default_runtime_conf_path,
                   setting_conf_prefix=file_utils.get_federatedml_setting_conf_directory(),
                   mode=job_type)
    return dsl_parser


def federated_order_reset(dest_partys, scheduler_partys_info):
    dest_partys_new = []
    scheduler = []
    dest_party_ids_dict = {}
    for dest_role, dest_party_ids in dest_partys:
        from copy import deepcopy
        new_dest_party_ids = deepcopy(dest_party_ids)
        dest_party_ids_dict[dest_role] = new_dest_party_ids
        for scheduler_role, scheduler_party_id in scheduler_partys_info:
            if dest_role == scheduler_role and scheduler_party_id in dest_party_ids:
                dest_party_ids_dict[dest_role].remove(scheduler_party_id)
                scheduler.append((scheduler_role, [scheduler_party_id]))
        if dest_party_ids_dict[dest_role]:
            dest_partys_new.append((dest_role, dest_party_ids_dict[dest_role]))
    if scheduler:
        dest_partys_new.extend(scheduler)
    return dest_partys_new


def get_parser_version_mapping():
    return {
        "1": DSLParser(),
        "2": DSLParserV2()
    }


def get_dsl_parser_by_version(version: str = "1"):
    mapping = get_parser_version_mapping()
    if isinstance(version, int):
        version = str(version)
    if version not in mapping:
        raise Exception("{} version of dsl parser is not currently supported.".format(version))
    return mapping[version]
