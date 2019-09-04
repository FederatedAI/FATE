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


def pipeline_dag_dependency(job_id, party_id, role):
    try:
        jobs = job_utils.query_job(job_id=job_id)
        if not jobs:
            raise Exception('query job {} failed'.format(job_id))
        job = jobs[0]
        job_dsl_parser = job_utils.get_job_dsl_parser(dsl=json_loads(job.f_dsl),
                                                      runtime_conf=json_loads(job.f_runtime_conf),
                                                      train_runtime_conf=json_loads(job.f_train_runtime_conf))
        dag_dependency = job_dsl_parser.get_dependency()
        roles = json_loads(job.f_roles)
        if role in roles:
            if party_id in roles[role]:
                party_index = roles[role].index(party_id)

                return dag_dependency[role][party_index]

            else:
                stat_logger.exception("party_id {} no found".format(party_id))
                raise "party_id {} no found".format(party_id)
        else:
            stat_logger.exception("role {} no found".format(role))
            raise "role {} no found".format(role)

    except Exception as e:
        stat_logger.exception(e)
        raise e
