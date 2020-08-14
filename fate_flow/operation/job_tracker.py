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
import uuid
import operator
from typing import List

from fate_arch.common.base_utils import current_timestamp, serialize_b64, deserialize_b64
from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import DB, TrackingMetric, TrackingOutputDataInfo, ComponentSummary
from fate_flow.entity.constant import Backend
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.model_manager import pipelined_model
from fate_arch import storage
from fate_flow.utils import job_utils, model_utils
from fate_arch.abc import StorageSessionABC


class Tracker(object):
    """
    Tracker for Job/TaskSet/Task/Metric
    """
    METRIC_DATA_PARTITION = 48
    METRIC_LIST_PARTITION = 48
    JOB_VIEW_PARTITION = 8

    def __init__(self, job_id: str, role: str, party_id: int,
                 model_id: str = None,
                 model_version: str = None,
                 task_set_id: int = None,
                 component_name: str = None,
                 component_module_name: str = None,
                 task_id: str = None,
                 task_version: int = None
                 ):
        self.job_id = job_id
        self.role = role
        self.party_id = party_id
        self.model_id = model_id
        self.party_model_id = model_utils.gen_party_model_id(model_id=model_id, role=role, party_id=party_id)
        self.model_version = model_version
        self.pipelined_model = None
        if self.party_model_id and self.model_version:
            self.pipelined_model = pipelined_model.PipelinedModel(model_id=self.party_model_id,
                                                                  model_version=self.model_version)

        self.task_set_id = task_set_id

        self.component_name = component_name if component_name else self.job_virtual_component_name()
        self.module_name = component_module_name if component_module_name else self.job_virtual_component_module_name()
        self.task_id = task_id
        self.task_version = task_version

    def save_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric], job_level=False):
        schedule_logger(self.job_id).info(
            'save job {} component {} on {} {} {} {} metric data'.format(self.job_id, self.component_name, self.role,
                                                                         self.party_id, metric_namespace, metric_name))
        kv = []
        for metric in metrics:
            kv.append((metric.key, metric.value))
        self.insert_metrics_into_db(metric_namespace, metric_name, 1, kv, job_level)

    def get_job_metric_data(self, metric_namespace: str, metric_name: str):
        return self.read_metric_data(metric_namespace=metric_namespace, metric_name=metric_name, job_level=True)

    def get_metric_data(self, metric_namespace: str, metric_name: str):
        return self.read_metric_data(metric_namespace=metric_namespace, metric_name=metric_name, job_level=False)

    def read_metric_data(self, metric_namespace: str, metric_name: str, job_level=False):
        with DB.connection_context():
            metrics = []
            for k, v in self.read_metrics_from_db(metric_namespace, metric_name, 1, job_level):
                metrics.append(Metric(key=k, value=v))
            return metrics

    def save_metric_meta(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta,
                         job_level: bool = False):
        schedule_logger(self.job_id).info(
            'save job {} component {} on {} {} {} {} metric meta'.format(self.job_id, self.component_name, self.role,
                                                                         self.party_id, metric_namespace, metric_name))
        self.insert_metrics_into_db(metric_namespace, metric_name, 0, metric_meta.to_dict().items(), job_level)

    def get_metric_meta(self, metric_namespace: str, metric_name: str, job_level: bool = False):
        with DB.connection_context():
            kv = dict()
            for k, v in self.read_metrics_from_db(metric_namespace, metric_name, 0, job_level):
                kv[k] = v
            return MetricMeta(name=kv.get('name'), metric_type=kv.get('metric_type'), extra_metas=kv)

    def log_job_view(self, view_data: dict):
        self.insert_metrics_into_db('job', 'job_view', 2, view_data.items(), job_level=True)

    def get_job_view(self):
        with DB.connection_context():
            view_data = {}
            for k, v in self.read_metrics_from_db('job', 'job_view', 2, job_level=True):
                view_data[k] = v
            return view_data

    def save_component_summary(self, summary_data: dict):
        with DB.connection_context():
            component_summary = ComponentSummary.select().where(ComponentSummary.f_job_id == self.job_id,
                                                                ComponentSummary.f_role == self.role,
                                                                ComponentSummary.f_party_id == self.party_id,
                                                                ComponentSummary.f_component_name == self.component_name)
            is_insert = True
            if component_summary:
                cpn_summary = component_summary[0]
                is_insert = False
            else:
                cpn_summary = ComponentSummary()
                cpn_summary.f_create_time = current_timestamp()
            cpn_summary.f_job_id = self.job_id
            cpn_summary.f_role = self.role
            cpn_summary.f_party_id = self.party_id
            cpn_summary.f_component_name = self.component_name
            cpn_summary.f_update_time = current_timestamp()
            cpn_summary.f_summary = serialize_b64(summary_data, to_str=True)

            if is_insert:
                cpn_summary.save(force_insert=True)
            else:
                cpn_summary.save()
            return cpn_summary

    def get_component_summary(self):
        with DB.connection_context():
            component_summary = ComponentSummary.select().where(ComponentSummary.f_job_id == self.job_id,
                                                                ComponentSummary.f_role == self.role,
                                                                ComponentSummary.f_party_id == self.party_id,
                                                                ComponentSummary.f_component_name == self.component_name)
            if component_summary:
                cpn_summary = component_summary[0]
                return deserialize_b64(cpn_summary.f_summary)
            else:
                return ""

    def save_output_data(self, data_table, output_storage_engine=None):
        if data_table:
            persistent_table_namespace, persistent_table_name = 'output_data_{}'.format(
                self.task_id), uuid.uuid1().hex
            schedule_logger(self.job_id).info(
                'persisting the component output temporary table to {} {}'.format(persistent_table_namespace,
                                                                                  persistent_table_name))
            partitions = data_table.partitions
            schedule_logger(self.job_id).info('output data table partitions is {}'.format(partitions))
            address = StorageSessionABC.register(name=persistent_table_name,
                                                 namespace=persistent_table_namespace,
                                                 storage_engine=output_storage_engine,
                                                 partitions=partitions)
            schema = {}
            data_table.save(address, schema=schema, partitions=partitions)
            table = storage.Session.build().get_table(name=persistent_table_name, namespace=persistent_table_namespace)
            part_of_data = []
            count = 100
            for k, v in data_table.collect():
                part_of_data.append((k, v))
                count -= 1
                if count == 0:
                    break
            table.update_metas(schema=schema, part_of_data=part_of_data, count=data_table.count(), partitions=partitions)
            return persistent_table_namespace, persistent_table_name
        else:
            schedule_logger(self.job_id).info('task id {} output data table is none'.format(self.task_id))
            return None, None

    def get_output_data_table(self, output_data_infos, init_session=False, need_all=True, session_id=''):
        """
        Get component output data table, will run in the task executor process
        :param data_name:
        :return:
        """
        if not init_session and not session_id:
            session_id = job_utils.generate_session_id(self.task_id, self.task_version, self.role, self.party_id)
        data_tables = {}
        if output_data_infos:
            for output_data_info in output_data_infos:
                schedule_logger(self.job_id).info("Get task {} {} output table {} {}".format(output_data_info.f_task_id, output_data_info.f_task_version, output_data_info.f_table_namespace, output_data_info.f_table_name))
                if not need_all:
                    data_table = StorageTable(name=output_data_info.f_table_name, namespace=output_data_info.f_table_namespace, data_name=output_data_info.f_data_name)
                else:
                    #data_table = storage.Session.build(name=output_data_info.f_table_name, namespace=output_data_info.f_table_namespace).get_table(name=output_data_info.f_table_name, namespace=output_data_info.f_table_namespace)
                    data_table = storage.StorageTableBase()
                data_tables[output_data_info.f_data_name] = data_table
        return data_tables

    def init_pipelined_model(self):
        self.pipelined_model.create_pipelined_model()

    def save_output_model(self, model_buffers: dict, model_alias: str):
        if model_buffers:
            self.pipelined_model.save_component_model(component_name=self.component_name,
                                                      component_module_name=self.module_name,
                                                      model_alias=model_alias,
                                                      model_buffers=model_buffers)

    def get_output_model(self, model_alias):
        model_buffers = self.pipelined_model.read_component_model(component_name=self.component_name,
                                                                  model_alias=model_alias)
        return model_buffers

    def collect_model(self):
        model_buffers = self.pipelined_model.collect_models()
        return model_buffers

    def save_pipelined_model(self, pipelined_buffer_object):
        self.save_output_model({'Pipeline': pipelined_buffer_object}, 'pipeline')
        self.pipelined_model.save_pipeline(pipelined_buffer_object=pipelined_buffer_object)

    def get_component_define(self):
        return self.pipelined_model.get_component_define(component_name=self.component_name)

    def insert_metrics_into_db(self, metric_namespace: str, metric_name: str, data_type: int, kv, job_level=False):
        with DB.connection_context():
            try:
                tracking_metric = self.get_dynamic_db_model(TrackingMetric, self.job_id)()
                tracking_metric.f_job_id = self.job_id
                tracking_metric.f_component_name = (self.component_name if not job_level else self.job_virtual_component_name())
                tracking_metric.f_task_id = self.task_id
                tracking_metric.f_task_version = self.task_version
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
                self.bulk_insert_into_db(self.get_dynamic_db_model(TrackingMetric, self.job_id),
                                         tracking_metric_data_source)
            except Exception as e:
                schedule_logger(self.job_id).exception("An exception where inserted metric {} of metric namespace: {} to database:\n{}".format(
                    metric_name,
                    metric_namespace,
                    e
                ))

    def log_output_data_info(self, data_name: str, table_namespace: str, table_name: str):
        self.insert_output_data_info_into_db(data_name=data_name, table_namespace=table_namespace, table_name=table_name)

    def insert_output_data_info_into_db(self, data_name: str, table_namespace: str, table_name: str):
        with DB.connection_context():
            try:
                tracking_output_data_info = self.get_dynamic_db_model(TrackingOutputDataInfo, self.job_id)()
                tracking_output_data_info.f_job_id = self.job_id
                tracking_output_data_info.f_component_name = self.component_name
                tracking_output_data_info.f_task_id = self.task_id
                tracking_output_data_info.f_task_version = self.task_version
                tracking_output_data_info.f_data_name = data_name
                tracking_output_data_info.f_role = self.role
                tracking_output_data_info.f_party_id = self.party_id
                tracking_output_data_info.f_table_namespace = table_namespace
                tracking_output_data_info.f_table_name = table_name
                tracking_output_data_info.f_create_time = current_timestamp()
                self.bulk_insert_into_db(self.get_dynamic_db_model(TrackingOutputDataInfo, self.job_id),
                                         [tracking_output_data_info.to_json()])
            except Exception as e:
                schedule_logger(self.job_id).exception("An exception where inserted output data info {} {} {} to database:\n{}".format(
                    data_name,
                    table_namespace,
                    table_name,
                    e
                ))

    def bulk_insert_into_db(self, model, data_source):
        with DB.connection_context():
            try:
                DB.create_tables([model])
                batch_size = 50 if RuntimeConfig.USE_LOCAL_DATABASE else 1000
                for i in range(0, len(data_source), batch_size):
                    with DB.atomic():
                        model.insert_many(data_source[i:i+batch_size]).execute()
                return len(data_source)
            except Exception as e:
                schedule_logger(self.job_id).exception(e)
                return 0

    def read_metrics_from_db(self, metric_namespace: str, metric_name: str, data_type, job_level=False):
        with DB.connection_context():
            metrics = []
            try:
                tracking_metric_model = self.get_dynamic_db_model(TrackingMetric, self.job_id)
                tracking_metrics = tracking_metric_model.select(tracking_metric_model.f_key, tracking_metric_model.f_value).where(
                    tracking_metric_model.f_job_id == self.job_id,
                    tracking_metric_model.f_component_name == (self.component_name if not job_level else self.job_virtual_component_name()),
                    tracking_metric_model.f_role == self.role,
                    tracking_metric_model.f_party_id == self.party_id,
                    tracking_metric_model.f_metric_namespace == metric_namespace,
                    tracking_metric_model.f_metric_name == metric_name,
                    tracking_metric_model.f_type == data_type
                )
                for tracking_metric in tracking_metrics:
                    yield deserialize_b64(tracking_metric.f_key), deserialize_b64(tracking_metric.f_value)
            except Exception as e:
                schedule_logger(self.job_id).exception(e)
                raise e
            return metrics

    def get_metric_list(self, job_level: bool = False):
        with DB.connection_context():
            metrics = dict()
            tracking_metric_model = self.get_dynamic_db_model(TrackingMetric, self.job_id)
            tracking_metrics = tracking_metric_model.select(tracking_metric_model.f_metric_namespace, tracking_metric_model.f_metric_name).where(
                                    tracking_metric_model.f_job_id==self.job_id,
                                    tracking_metric_model.f_component_name==(self.component_name if not job_level else 'dag'),
                                    tracking_metric_model.f_role==self.role,
                                    tracking_metric_model.f_party_id==self.party_id).distinct()
            for tracking_metric in tracking_metrics:
                metrics[tracking_metric.f_metric_namespace] = metrics.get(tracking_metric.f_metric_namespace, [])
                metrics[tracking_metric.f_metric_namespace].append(tracking_metric.f_metric_name)
            return metrics

    def get_output_data_info(self, data_name=None):
        return self.read_output_data_info_from_db(data_name=data_name)

    def read_output_data_info_from_db(self, data_name=None):
        filter_dict = {}
        filter_dict["job_id"] = self.job_id
        filter_dict["component_name"] = self.component_name
        filter_dict["role"] = self.role
        filter_dict["party_id"] = self.party_id
        if data_name:
            filter_dict["data_name"] = data_name
        return self.query_output_data_infos(**filter_dict)

    @classmethod
    def query_output_data_infos(cls, **kwargs):
        with DB.connection_context():
            tracking_output_data_info_model = cls.get_dynamic_db_model(TrackingOutputDataInfo, kwargs.get("job_id"))
            filters = []
            for f_n, f_v in kwargs.items():
                attr_name = 'f_%s' % f_n
                if hasattr(tracking_output_data_info_model, attr_name):
                    filters.append(operator.attrgetter('f_%s' % f_n)(tracking_output_data_info_model) == f_v)
            if filters:
                output_data_infos_tmp = tracking_output_data_info_model.select().where(*filters)
            else:
                output_data_infos_tmp = tracking_output_data_info_model.select()
            output_data_infos_group = {}
            # Only the latest version of the task output data is retrieved
            for output_data_info in output_data_infos_tmp:
                group_key = cls.get_output_data_group_key(output_data_info.f_task_id, output_data_info.f_data_name)
                if group_key not in output_data_infos_group:
                    output_data_infos_group[group_key] = output_data_info
                elif output_data_info.f_task_version > output_data_infos_group[group_key].f_task_version:
                    output_data_infos_group[group_key] = output_data_info
            return output_data_infos_group.values()

    @classmethod
    def get_output_data_group_key(cls, task_id, data_name):
        return task_id + data_name

    def clean_task(self, roles):
        schedule_logger(self.job_id).info('clean task {} on {} {}'.format(self.task_id,
                                                                          self.role,
                                                                          self.party_id))
        if Backend.EGGROLL != RuntimeConfig.BACKEND:
            return
        try:
            for role, party_ids in roles.items():
                for party_id in party_ids:
                    # clean up temporary tables
                    pass
                    '''
                    namespace_clean = job_utils.generate_session_id(task_id=self.task_id,
                                                                    task_version=self.task_version,
                                                                    role=role,
                                                                    party_id=party_id)
                    session.clean_tables(namespace=namespace_clean, regex_string='*')
                    schedule_logger(self.job_id).info('clean table by namespace {} on {} {} done'.format(namespace_clean,
                                                                                                         self.role,
                                                                                                         self.party_id))
                    # clean up the last tables of the federation
                    namespace_clean = job_utils.generate_federated_id(self.task_id, self.task_version)
                    session.clean_tables(namespace=namespace_clean, regex_string='*')
                    schedule_logger(self.job_id).info('clean table by namespace {} on {} {} done'.format(namespace_clean,
                                                                                                         self.role,
                                                                                                         self.party_id))
                    '''

        except Exception as e:
            schedule_logger(self.job_id).exception(e)
        schedule_logger(self.job_id).info('clean task {} on {} {} done'.format(self.task_id,
                                                                               self.role,
                                                                               self.party_id))

    @classmethod
    def get_dynamic_db_model(cls, base, job_id):
        return type(base.model(table_index=cls.get_dynamic_tracking_table_index(job_id=job_id)))

    @classmethod
    def get_dynamic_tracking_table_index(cls, job_id):
        return job_id[:8]

    @staticmethod
    def job_view_table_name():
        return '_'.join(['job', 'view'])

    @classmethod
    def job_virtual_component_name(cls):
        return "pipeline"

    @classmethod
    def job_virtual_component_module_name(cls):
        return "Pipeline"

