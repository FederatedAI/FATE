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
from arch.api.utils.core import json_loads
from fate_flow.settings import stat_logger
from fate_flow.utils import job_utils


def pipeline_dag_dependency(job_info):
    try:
        if job_info.get('job_id'):
            jobs = job_utils.query_job(job_id=job_info.get('job_id', ''))
            if not jobs:
                raise Exception('query job {} failed'.format(job_info.get('job_id', '')))
            job = jobs[0]
            job_dsl_parser = job_utils.get_job_dsl_parser(dsl=json_loads(job.f_dsl),
                                                          runtime_conf=json_loads(job.f_runtime_conf),
                                                          train_runtime_conf=json_loads(job.f_train_runtime_conf))
        else:
            job_dsl_parser = job_utils.get_job_dsl_parser(dsl=job_info.get('job_dsl', {}),
                                                          runtime_conf=job_info.get('job_runtime_conf', {}),
                                                          train_runtime_conf=job_info.get('job_train_runtime_conf', {}))
        return job_dsl_parser.get_dependency(role=job_info.get('role', ''), party_id=job_info.get('party_id', ''))
    except Exception as e:
        stat_logger.exception(e)
        raise e
