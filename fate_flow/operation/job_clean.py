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

import operator
from fate_arch.common.base_utils import current_timestamp
from fate_arch.common import base_utils
from fate_flow.db.db_models import DB, Job, Task
from fate_flow.entity.types import StatusSet, JobStatus, TaskStatus, EndStatus
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_arch.common.log import schedule_logger, sql_logger
import peewee


class JobClean(object):
    @classmethod
    def clean_table(job_id, role, party_id, component_name):
        # clean data table
        stat_logger.info('start delete {} {} {} {} data table'.format(job_id, role, party_id, component_name))

        tracker = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=component_name)
        output_data_table_infos = tracker.get_output_data_info()
        if output_data_table_infos:
            delete_tables_by_table_infos(output_data_table_infos)
            stat_logger.info('delete {} {} {} {} data table success'.format(job_id, role, party_id, component_name))


    @classmethod
    def start_clean_job(**kwargs):
        tasks = query_task(**kwargs)
        if tasks:
            for task in tasks:
                task_info = get_task_info(task.f_job_id, task.f_role, task.f_party_id, task.f_component_name)
                try:
                    # clean session
                    stat_logger.info('start {} {} {} {} session stop'.format(task.f_job_id, task.f_role,
                                                                             task.f_party_id, task.f_component_name))
                    start_session_stop(task)
                    stat_logger.info('stop {} {} {} {} session success'.format(task.f_job_id, task.f_role,
                                                                               task.f_party_id, task.f_component_name))
                except Exception as e:
                    pass
                try:
                    # clean data table
                    clean_table(job_id=task.f_job_id, role=task.f_role, party_id=task.f_party_id,
                                component_name=task.f_component_name)
                except Exception as e:
                    stat_logger.info('delete {} {} {} {} data table failed'.format(task.f_job_id, task.f_role,
                                                                                   task.f_party_id, task.f_component_name))
                    stat_logger.exception(e)
                try:
                    # clean metric data
                    stat_logger.info('start delete {} {} {} {} metric data'.format(task.f_job_id, task.f_role,
                                                                                   task.f_party_id, task.f_component_name))
                    delete_metric_data(task_info)
                    stat_logger.info('delete {} {} {} {} metric data success'.format(task.f_job_id, task.f_role,
                                                                                     task.f_party_id,
                                                                                     task.f_component_name))
                except Exception as e:
                    stat_logger.info('delete {} {} {} {} metric data failed'.format(task.f_job_id, task.f_role,
                                                                                    task.f_party_id,
                                                                                    task.f_component_name))
                    stat_logger.exception(e)
        else:
            raise Exception('no found task')
