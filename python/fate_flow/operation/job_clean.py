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
from fate_flow.manager.data_manager import delete_tables_by_table_infos, delete_metric_data
from fate_flow.operation.job_tracker import Tracker
from fate_flow.operation.job_saver import JobSaver
from fate_flow.settings import stat_logger
from fate_flow.utils.job_utils import start_session_stop


class JobClean(object):
    @classmethod
    def clean_table(cls, job_id, role, party_id, component_name):
        # clean data table
        stat_logger.info('start delete {} {} {} {} data table'.format(job_id, role, party_id, component_name))

        tracker = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=component_name)
        output_data_table_infos = tracker.get_output_data_info()
        if output_data_table_infos:
            delete_tables_by_table_infos(output_data_table_infos)
            stat_logger.info('delete {} {} {} {} data table success'.format(job_id, role, party_id, component_name))

    @classmethod
    def start_clean_job(cls, **kwargs):
        tasks = JobSaver.query_task(**kwargs)
        if tasks:
            for task in tasks:
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
                    JobClean.clean_table(job_id=task.f_job_id, role=task.f_role, party_id=task.f_party_id,
                                         component_name=task.f_component_name)
                except Exception as e:
                    stat_logger.info('delete {} {} {} {} data table failed'.format(task.f_job_id, task.f_role,
                                                                                   task.f_party_id,
                                                                                   task.f_component_name))
                    stat_logger.exception(e)
                try:
                    # clean metric data
                    stat_logger.info('start delete {} {} {} {} metric data'.format(task.f_job_id, task.f_role,
                                                                                   task.f_party_id,
                                                                                   task.f_component_name))
                    delete_metric_data({'job_id': task.f_job_id,
                                        'role': task.f_role,
                                        'party_id': task.f_party_id,
                                        'component_name': task.f_component_name})
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
