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
import base64
import pickle
from typing import List

from arch.api.utils import dtable_utils
from arch.api.utils.core import current_timestamp
from fate_flow.db.db_models import DB, Job, Task, TrackingMetric
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.manager import model_manager
from fate_flow.settings import stat_logger
from fate_flow.storage.fate_storage import FateStorage
from fate_flow.utils import job_utils


class Tracking(object):
    METRIC_DATA_PARTITION = 48
    METRIC_LIST_PARTITION = 48
    JOB_VIEW_PARTITION = 8

    def __init__(self, job_id: str, role: str, party_id: int, model_key: str = None, component_name: str = None,
                 task_id: str = None):
        self.job_id = job_id
        self.role = role
        self.party_id = party_id
        self.component_name = component_name if component_name else 'pipeline'
        self.task_id = task_id if task_id else ''
        self.table_namespace = '_'.join(
            ['fate_flow', 'tracking', 'data', self.job_id, self.role, str(self.party_id), self.component_name])
        self.job_table_namespace = '_'.join(
            ['fate_flow', 'tracking', 'data', self.job_id, self.role, str(self.party_id)])
        self.model_id = Tracking.gen_party_model_id(model_key=model_key, role=role, party_id=party_id)
        self.model_version = self.job_id

    def log_job_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        # TODO: In the next version will be changed to call the API way by the server persistent storage, not here to do
        self.save_metric_data(metric_namespace=metric_namespace, metric_name=metric_name, metrics=metrics,
                              job_level=True)

    def log_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        # TODO: In the next version will be changed to call the API way by the server persistent storage, not here to do
        stat_logger.info(
            'log job {} component {} on {} {} {} {} metric data'.format(self.job_id, self.component_name, self.role,
                                                                        self.party_id, metric_namespace, metric_name))
        self.save_metric_data(metric_namespace=metric_namespace, metric_name=metric_name, metrics=metrics,
                              job_level=False)

    def save_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric], job_level=False):
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
        # TODO: In the next version will be changed to call the API way by the server persistent storage, not here to do
        stat_logger.info(
            'set job {} component {} on {} {} {} {} metric meta'.format(self.job_id, self.component_name, self.role,
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
            stat_logger.info(query_sql)
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
            FateStorage.save_data_table_meta(
                {'schema': data_table.schema, 'header': data_table.schema.get('header', [])},
                namespace=persistent_table._namespace, name=persistent_table._name)
            data_table_info = {
                data_name: {'name': persistent_table._name, 'namespace': persistent_table._namespace}}
        else:
            data_table_info = {}
        FateStorage.save_data(
            data_table_info.items(),
            name=Tracking.output_table_name('data'),
            namespace=self.table_namespace,
            partition=48)

    def get_output_data_table(self, data_name: str = 'component'):
        output_data_info_table = FateStorage.table(name=Tracking.output_table_name('data'),
                                                   namespace=self.table_namespace)
        data_table_info = output_data_info_table.get(data_name)
        if data_table_info:
            data_table = FateStorage.table(name=data_table_info.get('name', ''),
                                           namespace=data_table_info.get('namespace', ''))
            data_table_meta = FateStorage.get_data_table_meta_by_instance(data_table=data_table)
            if data_table_meta.get('schema', None):
                data_table.schema = data_table_meta['schema']
            return data_table
        else:
            return None

    def save_output_model(self, model_buffers: dict, module_name: str):
        if model_buffers:
            model_manager.save_model(model_key=self.component_name,
                                     model_buffers=model_buffers,
                                     model_version=self.model_version,
                                     model_id=self.model_id)
            self.save_output_model_meta({'{}_module_name'.format(self.component_name): module_name})

    def get_output_model(self):
        model_buffers = model_manager.read_model(model_key=self.component_name,
                                                 model_version=self.model_version,
                                                 model_id=self.model_id)
        return model_buffers

    def collect_model(self):
        model_buffers = model_manager.collect_model(model_version=self.model_version, model_id=self.model_id)
        return model_buffers

    def save_output_model_meta(self, kv: dict):
        model_manager.save_model_meta(kv=kv,
                                      model_version=self.model_version,
                                      model_id=self.model_id)

    def get_output_model_meta(self):
        return model_manager.get_model_meta(model_version=self.model_version,
                                            model_id=self.model_id)

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
                    db_source['f_key'] = Tracking.serialize_b64(k)
                    db_source['f_value'] = Tracking.serialize_b64(v)
                    db_source['f_create_time'] = current_timestamp()
                    tracking_metric_data_source.append(db_source)
                self.bulk_insert_model_data(TrackingMetric.model(table_index=self.get_table_index()),
                                            tracking_metric_data_source)
            except Exception as e:
                print(e)
                stat_logger.exception(e)

    def bulk_insert_model_data(self, model, data_source):
        with DB.connection_context():
            try:
                DB.create_tables([model])
                with DB.atomic():
                    model.insert_many(data_source).execute()
                return len(data_source)
            except Exception as e:
                print(e)
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
                stat_logger.info(query_sql)
                for row in cursor.fetchall():
                    yield Tracking.deserialize_b64(row[0]), Tracking.deserialize_b64(row[1])
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
            elif create:
                job = Job()
                job.f_create_time = current_timestamp()
            else:
                return None
            job.f_job_id = self.job_id
            job.f_role = role
            job.f_party_id = party_id
            if 'f_status' in job_info:
                if job.f_status in ['success', 'failed', 'partial', 'deleted']:
                    # Termination status cannot be updated
                    # TODO:
                    pass
            for k, v in job_info.items():
                if k in ['f_job_id', 'f_role', 'f_party_id'] or v == getattr(Job, k).default:
                    continue
                setattr(job, k, v)
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
                if task.f_status in ['success', 'failed', 'partial', 'deleted']:
                    # Termination status cannot be updated
                    # TODO:
                    pass
            for k, v in task_info.items():
                if k in ['f_job_id', 'f_component_name', 'f_task_id', 'f_role', 'f_party_id'] or v == getattr(Task,
                                                                                                              k).default:
                    continue
                setattr(task, k, v)
            if is_insert:
                task.save(force_insert=True)
            else:
                task.save()
            return task

    def clean_job(self):
        FateStorage.clean_job(namespace=self.job_id, regex_string='*')

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

    @staticmethod
    def gen_party_model_id(model_key, role, party_id):
        return dtable_utils.gen_namespace_by_key(namespace_key=model_key, role=role,
                                                 party_id=party_id) if model_key else None

    @staticmethod
    def serialize_b64(src):
        return base64.b64encode(pickle.dumps(src))

    @staticmethod
    def deserialize_b64(src):
        return pickle.loads(base64.b64decode(src))


if __name__ == '__main__':
    FateStorage.init_storage()
    tracker = Tracking(job_utils.generate_job_id(), 'guest', 10000, 'hetero_lr')
    metric_namespace = 'TRAIN'
    metric_name = 'LOSS0'
    tracker.log_metric_data(metric_namespace, metric_name, [Metric(1, 0.2), Metric(2, 0.3)])

    metrics = tracker.get_metric_data(metric_namespace, metric_name)
    for metric in metrics:
        print(metric.key, metric.value)

    tracker.set_metric_meta(metric_namespace, metric_name,
                            MetricMeta(name=metric_name, metric_type='LOSS', extra_metas={'BEST': 0.2}))
    metric_meta = tracker.get_metric_meta(metric_namespace, metric_name)
    print(metric_meta.name, metric_meta.metric_type, metric_meta.metas)
