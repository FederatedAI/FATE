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
from fate_flow.utils.config_adapter import JobSubmitConfigAdapter


@DB.connection_context()
def get_job_dsl_parser_by_job_id(job_id):
    jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf).where(Job.f_job_id == job_id)
    if jobs:
        job = jobs[0]
        job_dsl_parser = get_job_dsl_parser(dsl=job.f_dsl, runtime_conf=job.f_runtime_conf,
                                            train_runtime_conf=job.f_train_runtime_conf)
        return job_dsl_parser
    else:
        return None


def get_job_dsl_parser(dsl=None, runtime_conf=None, pipeline_dsl=None, train_runtime_conf=None):
    parser_version = str(runtime_conf.get('dsl_version', '1'))
    dsl_parser = get_dsl_parser_by_version(parser_version)
    default_runtime_conf_path = os.path.join(file_utils.get_python_base_directory(),
                                             *['federatedml', 'conf', 'default_runtime_conf'])
    setting_conf_path = os.path.join(file_utils.get_python_base_directory(), *['federatedml', 'conf', 'setting_conf'])
    job_type = JobSubmitConfigAdapter(runtime_conf).get_job_type()
    dsl_parser.run(dsl=dsl,
                   runtime_conf=runtime_conf,
                   pipeline_dsl=pipeline_dsl,
                   pipeline_runtime_conf=train_runtime_conf,
                   default_runtime_conf_prefix=default_runtime_conf_path,
                   setting_conf_prefix=setting_conf_path,
                   mode=job_type)
    return dsl_parser


def get_parser_version_mapping():
    return {
        "1": DSLParser(),
        "2": DSLParserV2()
    }


def get_dsl_parser_by_version(version: str = "1"):
    mapping = get_parser_version_mapping()
    if version not in mapping:
        raise Exception("{} version of dsl parser is not currently supported.".format(version))
    return mapping[version]
