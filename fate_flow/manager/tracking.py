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
from typing import List

from arch.api import session, WorkMode
from arch.api.utils.core import current_timestamp, serialize_b64, deserialize_b64
from fate_flow.db.db_models import DB, Job, Task, TrackingMetric, DataView
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.manager import model_manager
from fate_flow.settings import stat_logger, API_VERSION, MAX_CONCURRENT_JOB_RUN_HOST
from fate_flow.utils import job_utils, api_utils, model_utils
from fate_flow.entity.constant_config import JobStatus, TaskStatus
from fate_flow.entity.runtime_config import RuntimeConfig


class Tracking(object):
    METRIC_DATA_PARTITION = 48
    METRIC_LIST_PARTITION = 48
    JOB_VIEW_PARTITION = 8

    def __init__(self, job_id: str, role: str, party_id: int,
                 model_id: str = None,
                 model_version: str = None,
                 component_name: str = None,
                 module_name: str = None,
                 task_id: str = None):
        self.job_id = job_id
        self.role = role
        self.party_id = party_id
        self.component_name = component_name if component_name else 'pipeline'
        self.module_name = module_name if module_name else 'Pipeline'
        self.task_id = task_id if task_id else job_utils.generate_task_id(job_id=self.job_id,
                                                                          component_name=self.component_name)
        self.table_namespace = '_'.join(
            ['fate_flow', 'tracking', 'data', self.job_id, self.role, str(self.party_id), self.component_name])
        self.job_table_namespace = '_'.join(
            ['fate_flow', 'tracking', 'data', self.job_id, self.role, str(self.party_id)])
        self.model_id = model_id
        self.party_model_id = model_utils.gen_party_model_id(model_id=model_id, role=role, party_id=party_id)
        self.model_version = model_version

    def log_job_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        self.save_metric_data_remote(metric_namespace=metric_namespace, metric_name=metric_name, metrics=metrics,
                                     job_level=True)

    def log_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        self.save_metric_data_remote(metric_namespace=metric_namespace, metric_name=metric_name, metrics=metrics,
                                     job_level=False)

    def save_metric_data_remote(self, metric_namespace: str, metric_name: str, metrics: List[Metric], job_level=False):
        # TODO: In the next version will be moved to tracking api module on arch/api package
        stat_logger.info(
            'request save job {} component {} on {} {} {} {} metric data'.format(self.job_id, self.component_name,
                                                                                 self.role,
                                                                                 self.party_id, metric_namespace,
                                                                                 metric_name))
        request_body = dict()
        request_body['metric_namespace'] = metric_namespace
        request_body['metric_name'] = metric_name
        request_body['metrics'] = [serialize_b64(metric, to_str=True) for metric in metrics]
        request_body['job_level'] = job_level
        response = api_utils.local_api(method='POST',
                                       endpoint='/{}/tracking/{}/{}/{}/{}/{}/metric_data/save'.format(
                                           API_VERSION,
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        return response['retcode'] == 0

    def save_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric], job_level=False):
        stat_logger.info(
            'save job {} component {} on {} {} {} {} metric data'.format(self.job_id, self.component_name, self.role,
                                                                         self.party_id, metric_namespace, metric_name))
        kv = []
        for metric in metrics:
            kv.append((metric.key, metric.value))
        self.insert_data_to_db(metric_namespace, metric_name, 1, kv, job_level)

    def get_job_metric_data(self, metric_namespace: str, metric_name: str):
        return self.read_metric_data(metric_namespace=metric_namespace, metric_name=metric_name, job_level=True)

    def get_metric_data(self, metric_namespace: str, metric_name: str):
        return self.read_metric_data(metric_namespace=metric_namespace, metric_name=metric_name, job_level=False)

    def read_metric_data(self, metric_namespace: str, metric_name: str, job_level=False):
        with DB.connection_context():
            metrics = []
            for k, v in self.read_data_from_db(metric_namespace, metric_name, 1, job_level):
                metrics.append(Metric(key=k, value=v))
            return metrics

    def set_metric_meta(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta,
                        job_level: bool = False):
        self.save_metric_meta_remote(metric_namespace=metric_namespace, metric_name=metric_name,
                                     metric_meta=metric_meta, job_level=job_level)

    def save_metric_meta_remote(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta,
                                job_level: bool = False):
        # TODO: In the next version will be moved to tracking api module on arch/api package
        stat_logger.info(
            'request save job {} component {} on {} {} {} {} metric meta'.format(self.job_id, self.component_name,
                                                                                 self.role,
                                                                                 self.party_id, metric_namespace,
                                                                                 metric_name))
        request_body = dict()
        request_body['metric_namespace'] = metric_namespace
        request_body['metric_name'] = metric_name
        request_body['metric_meta'] = serialize_b64(metric_meta, to_str=True)
        request_body['job_level'] = job_level
        response = api_utils.local_api(method='POST',
                                       endpoint='/{}/tracking/{}/{}/{}/{}/{}/metric_meta/save'.format(
                                           API_VERSION,
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        return response['retcode'] == 0

    def save_metric_meta(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta,
                         job_level: bool = False):
        stat_logger.info(
            'save job {} component {} on {} {} {} {} metric meta'.format(self.job_id, self.component_name, self.role,
                                                                         self.party_id, metric_namespace, metric_name))
        self.insert_data_to_db(metric_namespace, metric_name, 0, metric_meta.to_dict().items(), job_level)

    def get_metric_meta(self, metric_namespace: str, metric_name: str, job_level: bool = False):
        with DB.connection_context():
            kv = dict()
            for k, v in self.read_data_from_db(metric_namespace, metric_name, 0, job_level):
                kv[k] = v
            return MetricMeta(name=kv.get('name'), metric_type=kv.get('metric_type'), extra_metas=kv)

    def get_metric_list(self, job_level: bool = False):
        with DB.connection_context():
            metrics = dict()
            query_sql = 'select distinct f_metric_namespace, f_metric_name from t_tracking_metric_{} where ' \
                        'f_job_id = "{}" and f_component_name = "{}" and f_role = "{}" and f_party_id = "{}" ' \
                        'and f_task_id = "{}"'.format(
                self.get_table_index(), self.job_id, self.component_name if not job_level else 'dag', self.role,
                self.party_id, self.task_id)
            cursor = DB.execute_sql(query_sql)
            for row in cursor.fetchall():
                metrics[row[0]] = metrics.get(row[0], [])
                metrics[row[0]].append(row[1])
            return metrics

    def log_job_view(self, view_data: dict):
        self.insert_data_to_db('job', 'job_view', 2, view_data.items(), job_level=True)

    def get_job_view(self):
        with DB.connection_context():
            view_data = {}
            for k, v in self.read_data_from_db('job', 'job_view', 2, job_level=True):
                view_data[k] = v
            return view_data

    def save_output_data_table(self, data_table, data_name: str = 'component'):
        if data_table:
            persistent_table = data_table.save_as(namespace=data_table._namespace,
                                                  name='{}_persistent'.format(data_table._name))
            session.save_data_table_meta(
                {'schema': data_table.schema, 'header': data_table.schema.get('header', [])},
                data_table_namespace=persistent_table._namespace, data_table_name=persistent_table._name)
            data_table_info = {
                data_name: {'name': persistent_table._name, 'namespace': persistent_table._namespace}}
        else:
            data_table_info = {}
        session.save_data(
            data_table_info.items(),
            name=Tracking.output_table_name('data'),
            namespace=self.table_namespace,
            partition=48)
        self.save_data_view(self.role, self.party_id,
                            data_info={'f_table_name': persistent_table._name if data_table else '',
                                       'f_table_namespace': persistent_table._namespace if data_table else '',
                                       'f_partition': persistent_table._partitions if data_table else None,
                                       'f_table_count_actual': data_table.count() if data_table else 0},
                            mark=True)

    def get_output_data_table(self, data_name: str = 'component'):
        output_data_info_table = session.table(name=Tracking.output_table_name('data'),
                                               namespace=self.table_namespace)
        data_table_info = output_data_info_table.get(data_name)
        if data_table_info:
            data_table = session.table(name=data_table_info.get('name', ''),
                                       namespace=data_table_info.get('namespace', ''))
            data_table_meta = data_table.get_metas()
            if data_table_meta.get('schema', None):
                data_table.schema = data_table_meta['schema']
            return data_table
        else:
            return None

    def save_output_model(self, model_buffers: dict, model_name: str):
        if model_buffers:
            model_manager.save_component_model(component_model_key='{}.{}'.format(self.component_name, model_name),
                                               model_buffers=model_buffers,
                                               party_model_id=self.party_model_id,
                                               model_version=self.model_version)
            self.save_data_view(self.role, self.party_id,
                                data_info={'f_party_model_id': self.party_model_id,
                                           'f_model_version': self.model_version})
            self.save_output_model_meta({'{}_module_name'.format(self.component_name): self.module_name})

    def get_output_model(self, model_name):
        model_buffers = model_manager.read_component_model(
            component_model_key='{}.{}'.format(self.component_name, model_name),
            party_model_id=self.party_model_id,
            model_version=self.model_version)

        return model_buffers

    def collect_model(self):
        model_buffers = model_manager.collect_pipeline_model(party_model_id=self.party_model_id,
                                                             model_version=self.model_version)
        return model_buffers

    def save_output_model_meta(self, kv: dict):
        model_manager.save_pipeline_model_meta(kv=kv,
                                               party_model_id=self.party_model_id,
                                               model_version=self.model_version)

    def get_output_model_meta(self):
        return model_manager.get_pipeline_model_meta(party_model_id=self.party_model_id,
                                                     model_version=self.model_version)

    def insert_data_to_db(self, metric_namespace: str, metric_name: str, data_type: int, kv, job_level=False):
        with DB.connection_context():
            try:
                tracking_metric = TrackingMetric.model(table_index=self.job_id)
                tracking_metric.f_job_id = self.job_id
                tracking_metric.f_component_name = self.component_name if not job_level else 'dag'
                tracking_metric.f_task_id = self.task_id
                tracking_metric.f_role = self.role
                tracking_metric.f_party_id = self.party_id
                tracking_metric.f_metric_namespace = metric_namespace
                tracking_metric.f_metric_name = metric_name
                tracking_metric.f_type = data_type
                default_db_source = tracking_metric.to_json()
                tracking_metric_data_source = []
                for k, v in kv:
                    db_source = default_db_source.copy()
                    db_source['f_key'] = serialize_b64(k)
                    db_source['f_value'] = serialize_b64(v)
                    db_source['f_create_time'] = current_timestamp()
                    tracking_metric_data_source.append(db_source)
                self.bulk_insert_model_data(TrackingMetric.model(table_index=self.get_table_index()),
                                            tracking_metric_data_source)
            except Exception as e:
                stat_logger.exception(e)

    def bulk_insert_model_data(self, model, data_source):
        with DB.connection_context():
            try:
                DB.create_tables([model])
                batch_size = 50 if RuntimeConfig.USE_LOCAL_DATABASE else 1000
                for i in range(0, len(data_source), batch_size):
                    with DB.atomic():
                        model.insert_many(data_source[i:i+batch_size]).execute()
                return len(data_source)
            except Exception as e:
                stat_logger.exception(e)
                return 0

    def read_data_from_db(self, metric_namespace: str, metric_name: str, data_type, job_level=False):
        with DB.connection_context():
            metrics = []
            try:
                query_sql = 'select f_key, f_value from t_tracking_metric_{} where ' \
                            'f_job_id = "{}" and f_component_name = "{}" and f_role = "{}" and f_party_id = "{}"' \
                            'and f_task_id = "{}" and f_metric_namespace = "{}" and f_metric_name= "{}" and f_type="{}" order by f_id'.format(
                    self.get_table_index(), self.job_id, self.component_name if not job_level else 'dag', self.role,
                    self.party_id, self.task_id, metric_namespace, metric_name, data_type)
                cursor = DB.execute_sql(query_sql)
                for row in cursor.fetchall():
                    yield deserialize_b64(row[0]), deserialize_b64(row[1])
            except Exception as e:
                stat_logger.exception(e)
            return metrics

    def save_job_info(self, role, party_id, job_info, create=False):
        with DB.connection_context():
            stat_logger.info('save {} {} job: {}'.format(role, party_id, job_info))
            jobs = Job.select().where(Job.f_job_id == self.job_id, Job.f_role == role, Job.f_party_id == party_id)
            is_insert = True
            if jobs:
                job = jobs[0]
                is_insert = False
                if job.f_status == JobStatus.TIMEOUT:
                    return None
            elif create:
                job = Job()
                job.f_create_time = current_timestamp()
            else:
                return None
            job.f_job_id = self.job_id
            job.f_role = role
            job.f_party_id = party_id
            if 'f_status' in job_info:
                if job.f_status in [JobStatus.COMPLETE, JobStatus.FAILED]:
                    # Termination status cannot be updated
                    # TODO:
                    pass
                if job_info['f_status'] == JobStatus.FAILED and not job.f_end_time:
                    job.f_end_time = current_timestamp()
                    job.f_elapsed = job.f_end_time - job.f_start_time
                    job.f_update_time = current_timestamp()
            for k, v in job_info.items():
                try:
                    if k in ['f_job_id', 'f_role', 'f_party_id'] or v == getattr(Job, k).default:
                        continue
                    setattr(job, k, v)
                except:
                    pass

            if is_insert:
                job.save(force_insert=True)
            else:
                job.save()

    def save_task(self, role, party_id, task_info):
        with DB.connection_context():
            tasks = Task.select().where(Task.f_job_id == self.job_id,
                                        Task.f_component_name == self.component_name,
                                        Task.f_task_id == self.task_id,
                                        Task.f_role == role,
                                        Task.f_party_id == party_id)
            is_insert = True
            if tasks:
                task = tasks[0]
                is_insert = False
            else:
                task = Task()
                task.f_create_time = current_timestamp()
            task.f_job_id = self.job_id
            task.f_component_name = self.component_name
            task.f_task_id = self.task_id
            task.f_role = role
            task.f_party_id = party_id
            if 'f_status' in task_info:
                if task.f_status in [TaskStatus.COMPLETE, TaskStatus.FAILED]:
                    # Termination status cannot be updated
                    # TODO:
                    pass
            for k, v in task_info.items():
                try:
                    if k in ['f_job_id', 'f_component_name', 'f_task_id', 'f_role', 'f_party_id'] or v == getattr(Task,
                                                                                                                  k).default:
                        continue
                except:
                    pass
                setattr(task, k, v)
            if is_insert:
                task.save(force_insert=True)
            else:
                task.save()
            return task

    def save_data_view(self, role, party_id, data_info, mark=False):
        with DB.connection_context():
            data_views = DataView.select().where(DataView.f_job_id == self.job_id,
                                                 DataView.f_component_name == self.component_name,
                                                 DataView.f_task_id == self.task_id,
                                                 DataView.f_role == role,
                                                 DataView.f_party_id == party_id)
            is_insert = True
            if mark and self.component_name == "upload_0":
                return
            if data_views:
                data_view = data_views[0]
                is_insert = False
            else:
                data_view = DataView()
                data_view.f_create_time = current_timestamp()
            data_view.f_job_id = self.job_id
            data_view.f_component_name = self.component_name
            data_view.f_task_id = self.task_id
            data_view.f_role = role
            data_view.f_party_id = party_id
            data_view.f_update_time = current_timestamp()
            for k, v in data_info.items():
                if k in ['f_job_id', 'f_component_name', 'f_task_id', 'f_role', 'f_party_id'] or v == getattr(
                        DataView, k).default:
                    continue
                setattr(data_view, k, v)
            if is_insert:
                data_view.save(force_insert=True)
            else:
                data_view.save()
            return data_view

    def clean_task(self, roles, party_ids):
        stat_logger.info('clean table by namespace {}'.format(self.task_id))
        session.clean_tables(namespace=self.task_id, regex_string='*')
        for role in roles.split(','):
            for party_id in party_ids.split(','):
                session.clean_tables(namespace=self.task_id + '_' + role + '_' + party_id, regex_string='*')


    def job_quantity_constraint(self):
        if RuntimeConfig.WORK_MODE == WorkMode.CLUSTER:
            if self.role == 'host':
                running_jobs = job_utils.query_job(status='running', role=self.role)
                if len(running_jobs) >= MAX_CONCURRENT_JOB_RUN_HOST:
                    raise Exception('The job running on the host side exceeds the maximum running amount')

    def get_table_namespace(self, job_level: bool = False):
        return self.table_namespace if not job_level else self.job_table_namespace

    def get_table_index(self):
        return self.job_id[:8]

    @staticmethod
    def metric_table_name(metric_namespace: str, metric_name: str):
        return '_'.join(['metric', metric_namespace, metric_name])

    @staticmethod
    def metric_list_table_name():
        return '_'.join(['metric', 'list'])

    @staticmethod
    def output_table_name(output_type: str):
        return '_'.join(['output', output_type])

    @staticmethod
    def job_view_table_name():
        return '_'.join(['job', 'view'])
