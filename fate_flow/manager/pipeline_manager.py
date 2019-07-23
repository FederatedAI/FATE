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
from fate_flow.db.db_models import DB, Job
from fate_flow.utils import job_utils


@DB.connection_context()
def pipeline_dag_dependency(job_id):
    jobs = Job.select(Job.f_dsl, Job.f_runtime_conf).where(Job.f_job_id == job_id, Job.f_is_initiator == 1)
    if jobs:
        job_dsl_path, job_runtime_conf_path = job_utils.get_job_conf_path(job_id=job_id)
        job_dsl_parser = job_utils.get_job_dsl_parser(job_dsl_path=job_dsl_path,
                                                      job_runtime_conf_path=job_runtime_conf_path)
        return job_dsl_parser.get_dependency()
    else:
        return None
