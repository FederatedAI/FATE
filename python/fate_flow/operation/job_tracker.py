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
import copy
from typing import List

from fate_arch.common import EngineType, Party
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.common.base_utils import current_timestamp, serialize_b64, deserialize_b64, json_loads
from fate_arch.common.log import schedule_logger
from fate_flow.db.db_models import (DB, Job, TrackingMetric, TrackingOutputDataInfo,
                                    ComponentSummary, MachineLearningModelInfo as MLModel)
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.pipelined_model import pipelined_model
from fate_arch import storage
from fate_flow.utils import model_utils, job_utils, data_utils
from fate_arch import session
from fate_flow.entity.types import RunParameters


class Tracker(object):
    """
    Tracker for Job/Task/Metric
    """
    METRIC_DATA_PARTITION = 48
    METRIC_LIST_PARTITION = 48
    JOB_VIEW_PARTITION = 8

    def __init__(self, job_id: str, role: str, party_id: int,
                 model_id: str = None,
                 model_version: str = None,
                 component_name: str = None,
                 component_module_name: str = None,
                 task_id: str = None,
                 task_version: int = None,
                 job_parameters: RunParameters = None
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

        self.component_name = component_name if component_name else job_utils.job_virtual_component_name()
        self.module_name = component_module_name if component_module_name else job_utils.job_virtual_component_module_name()
        self.task_id = task_id
        self.task_version = task_version
        self.job_parameters = job_parameters

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

    @DB.connection_context()
    def read_metric_data(self, metric_namespace: str, metric_name: str, job_level=False):
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

    @DB.connection_context()
    def get_metric_meta(self, metric_namespace: str, metric_name: str, job_level: bool = False):
        kv = dict()
        for k, v in self.read_metrics_from_db(metric_namespace, metric_name, 0, job_level):
            kv[k] = v
        return MetricMeta(name=kv.get('name'), metric_type=kv.get('metric_type'), extra_metas=kv)

    def log_job_view(self, view_data: dict):
        self.insert_metrics_into_db('job', 'job_view', 2, view_data.items(), job_level=True)

    @DB.connection_context()
    def get_job_view(self):
        view_data = {}
        for k, v in self.read_metrics_from_db('job', 'job_view', 2, job_level=True):
            view_data[k] = v
        return view_data

    def save_output_data(self, computing_table, output_storage_engine, output_storage_address: dict,
                         output_table_namespace=None, output_table_name=None):
        if computing_table:
            if not output_table_namespace or not output_table_name:
                output_table_namespace, output_table_name = data_utils.default_output_table_info(task_id=self.task_id, task_version=self.task_version)
            schedule_logger(self.job_id).info(
                'persisting the component output temporary table to {} {}'.format(output_table_namespace,
                                                                                  output_table_name))
            partitions = computing_table.partitions
            schedule_logger(self.job_id).info('output data table partitions is {}'.format(partitions))
            address_dict = output_storage_address.copy()
            if output_storage_engine == StorageEngine.EGGROLL:
                address_dict.update({"name": output_table_name, "namespace": output_table_namespace, "storage_type": storage.EggRollStorageType.ROLLPAIR_LMDB})
            elif output_storage_engine == StorageEngine.STANDALONE:
                address_dict.update({"name": output_table_name, "namespace": output_table_namespace, "storage_type": storage.StandaloneStorageType.ROLLPAIR_LMDB})
            elif output_storage_engine == StorageEngine.HDFS:
                address_dict.update({"path": data_utils.default_output_fs_path(name=output_table_name, namespace=output_table_namespace, prefix=address_dict.get("path_prefix"))})
            else:
                raise RuntimeError(f"{output_storage_engine} storage is not supported")
            address = storage.StorageTableMeta.create_address(storage_engine=output_storage_engine, address_dict=address_dict)
            schema = {}
            # persistent table
            computing_table.save(address, schema=schema, partitions=partitions)
            part_of_data = []
            part_of_limit = 100
            for k, v in computing_table.collect():
                part_of_data.append((k, v))
                part_of_limit -= 1
                if part_of_limit == 0:
                    break
            table_count = computing_table.count()
            table_meta = storage.StorageTableMeta(name=output_table_name, namespace=output_table_namespace, new=True)
            table_meta.address = address
            table_meta.partitions = computing_table.partitions
            table_meta.engine = output_storage_engine
            table_meta.type = storage.EggRollStorageType.ROLLPAIR_LMDB
            table_meta.schema = schema
            table_meta.part_of_data = part_of_data
            table_meta.count = table_count
            table_meta.create()
            return output_table_namespace, output_table_name
        else:
            schedule_logger(self.job_id).info('task id {} output data table is none'.format(self.task_id))
            return None, None

    def get_output_data_table(self, output_data_infos):
        """
        Get component output data table, will run in the task executor process
        :param output_data_infos:
        :return:
        """
        output_tables_meta = {}
        if output_data_infos:
            for output_data_info in output_data_infos:
                schedule_logger(self.job_id).info("Get task {} {} output table {} {}".format(output_data_info.f_task_id, output_data_info.f_task_version, output_data_info.f_table_namespace, output_data_info.f_table_name))
                data_table_meta = storage.StorageTableMeta(name=output_data_info.f_table_name, namespace=output_data_info.f_table_namespace)
                output_tables_meta[output_data_info.f_data_name] = data_table_meta
        return output_tables_meta

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

    @DB.connection_context()
    def insert_metrics_into_db(self, metric_namespace: str, metric_name: str, data_type: int, kv, job_level=False):
        try:
            tracking_metric = self.get_dynamic_db_model(TrackingMetric, self.job_id)()
            tracking_metric.f_job_id = self.job_id
            tracking_metric.f_component_name = (self.component_name if not job_level else job_utils.job_virtual_component_name())
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

    @DB.connection_context()
    def insert_summary_into_db(self, summary_data: dict):
        try:
            summary_model = self.get_dynamic_db_model(ComponentSummary, self.job_id)
            DB.create_tables([summary_model])
            summary_obj = summary_model.get_or_none(
                summary_model.f_job_id == self.job_id,
                summary_model.f_component_name == self.component_name,
                summary_model.f_role == self.role,
                summary_model.f_party_id == self.party_id,
                summary_model.f_task_id == self.task_id,
                summary_model.f_task_version == self.task_version
            )
            if summary_obj:
                summary_obj.f_summary = serialize_b64(summary_data, to_str=True)
                summary_obj.f_update_time = current_timestamp()
                summary_obj.save()
            else:
                self.get_dynamic_db_model(ComponentSummary, self.job_id).create(
                    f_job_id=self.job_id,
                    f_component_name=self.component_name,
                    f_role=self.role,
                    f_party_id=self.party_id,
                    f_task_id=self.task_id,
                    f_task_version=self.task_version,
                    f_summary=serialize_b64(summary_data, to_str=True),
                    f_create_time=current_timestamp()
                )
        except Exception as e:
            schedule_logger(self.job_id).exception("An exception where querying summary job id: {} "
                                                   "component name: {} to database:\n{}".format(
                self.job_id, self.component_name, e)
            )

    @DB.connection_context()
    def read_summary_from_db(self):
        try:
            summary_model = self.get_dynamic_db_model(ComponentSummary, self.job_id)
            summary = summary_model.get_or_none(
                summary_model.f_job_id == self.job_id,
                summary_model.f_component_name == self.component_name,
                summary_model.f_role == self.role,
                summary_model.f_party_id == self.party_id
            )
            if summary:
                cpn_summary = deserialize_b64(summary.f_summary)
            else:
                cpn_summary = ""
        except Exception as e:
            schedule_logger(self.job_id).exception(e)
            raise e
        return cpn_summary

    def log_output_data_info(self, data_name: str, table_namespace: str, table_name: str):
        self.insert_output_data_info_into_db(data_name=data_name, table_namespace=table_namespace, table_name=table_name)

    @DB.connection_context()
    def insert_output_data_info_into_db(self, data_name: str, table_namespace: str, table_name: str):
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

    @DB.connection_context()
    def bulk_insert_into_db(self, model, data_source):
        try:
            try:
                DB.create_tables([model])
            except Exception as e:
                schedule_logger(self.job_id).exception(e)
            batch_size = 50 if RuntimeConfig.USE_LOCAL_DATABASE else 1000
            for i in range(0, len(data_source), batch_size):
                with DB.atomic():
                    model.insert_many(data_source[i:i+batch_size]).execute()
            return len(data_source)
        except Exception as e:
            schedule_logger(self.job_id).exception(e)
            return 0

    def save_as_table(self, computing_table, name, namespace):
        self.save_output_data(computing_table=computing_table,
                              output_storage_engine=self.job_parameters.storage_engine,
                              output_storage_address=self.job_parameters.engines_address.get(EngineType.STORAGE, {}),
                              output_table_namespace=namespace, output_table_name=name)

    @DB.connection_context()
    def read_metrics_from_db(self, metric_namespace: str, metric_name: str, data_type, job_level=False):
        metrics = []
        try:
            tracking_metric_model = self.get_dynamic_db_model(TrackingMetric, self.job_id)
            tracking_metrics = tracking_metric_model.select(tracking_metric_model.f_key, tracking_metric_model.f_value).where(
                tracking_metric_model.f_job_id == self.job_id,
                tracking_metric_model.f_component_name == (self.component_name if not job_level else job_utils.job_virtual_component_name()),
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

    @DB.connection_context()
    def clean_metrics(self):
        tracking_metric_model = self.get_dynamic_db_model(TrackingMetric, self.job_id)
        operate = tracking_metric_model.delete().where(
            tracking_metric_model.f_task_id==self.task_id,
            tracking_metric_model.f_task_version==self.task_version,
            tracking_metric_model.f_role==self.role,
            tracking_metric_model.f_party_id==self.party_id
        )
        return operate.execute() > 0

    @DB.connection_context()
    def get_metric_list(self, job_level: bool = False):
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
    @DB.connection_context()
    def query_output_data_infos(cls, **kwargs):
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

    def clean_task(self, runtime_conf):
        schedule_logger(self.job_id).info('clean task {} {} on {} {}'.format(self.task_id,
                                                                             self.task_version,
                                                                             self.role,
                                                                             self.party_id))
        try:
            sess = session.Session(computing_type=self.job_parameters.computing_engine, federation_type=self.job_parameters.federation_engine)
            # clean up temporary tables
            computing_temp_namespace = job_utils.generate_session_id(task_id=self.task_id,
                                                                     task_version=self.task_version,
                                                                     role=self.role,
                                                                     party_id=self.party_id)
            if self.job_parameters.computing_engine == ComputingEngine.EGGROLL:
                session_options = {"eggroll.session.processors.per.node": 1}
            else:
                session_options = {}
            sess.init_computing(computing_session_id=f"{computing_temp_namespace}_clean", options=session_options)
            sess.computing.cleanup(namespace=computing_temp_namespace, name="*")
            schedule_logger(self.job_id).info('clean table by namespace {} on {} {} done'.format(computing_temp_namespace,
                                                                                                 self.role,
                                                                                                 self.party_id))
            # clean up the last tables of the federation
            federation_temp_namespace = job_utils.generate_task_version_id(self.task_id, self.task_version)
            sess.computing.cleanup(namespace=federation_temp_namespace, name="*")
            schedule_logger(self.job_id).info('clean table by namespace {} on {} {} done'.format(federation_temp_namespace,
                                                                                                 self.role,
                                                                                                 self.party_id))
            sess.computing.stop()
            if self.job_parameters.federation_engine == FederationEngine.RABBITMQ and self.role != "local":
                schedule_logger(self.job_id).info('rabbitmq start clean up')
                parties = [Party(k, p) for k, v in runtime_conf['role'].items() for p in v]
                federation_session_id = job_utils.generate_task_version_id(self.task_id, self.task_version)
                component_parameters_on_party = copy.deepcopy(runtime_conf)
                component_parameters_on_party["local"] = {"role": self.role, "party_id": self.party_id}
                sess.init_federation(federation_session_id=federation_session_id,
                                     runtime_conf=component_parameters_on_party,
                                     service_conf=self.job_parameters.engines_address.get(EngineType.FEDERATION, {}))
                sess._federation_session.cleanup(parties)
                schedule_logger(self.job_id).info('rabbitmq clean up success')

            #TODO optimize the clean process
            if self.job_parameters.federation_engine == FederationEngine.PULSAR and self.role != "local": 
                schedule_logger(self.job_id).info('start to clean up pulsar topics')
                parties = [Party(k, p) for k, v in runtime_conf['role'].items() for p in v]
                federation_session_id = job_utils.generate_task_version_id(self.task_id, self.task_version)
                component_parameters_on_party = copy.deepcopy(runtime_conf)
                component_parameters_on_party["local"] = {"role": self.role, "party_id": self.party_id}
                sess.init_federation(federation_session_id=federation_session_id,
                                     runtime_conf=component_parameters_on_party,
                                     service_conf=self.job_parameters.engines_address.get(EngineType.FEDERATION, {}))
                sess._federation_session.cleanup(parties)
                schedule_logger(self.job_id).info('pulsar topic clean up success')
                
            return True
        except Exception as e:
            schedule_logger(self.job_id).exception(e)
            return False

    @DB.connection_context()
    def save_machine_learning_model_info(self):
        try:
            record = MLModel.get_or_none(MLModel.f_model_version == self.job_id,
                                         MLModel.f_role == self.role,
                                         MLModel.f_model_id == self.model_id,
                                         MLModel.f_party_id == self.party_id)
            if not record:
                job = Job.get_or_none(Job.f_job_id == self.job_id)
                pipeline = self.pipelined_model.read_component_model('pipeline', 'pipeline')['Pipeline']
                if job:
                    job_data = job.to_json()
                    model_info = {
                        'job_id': job_data.get("f_job_id"),
                        'role': self.role,
                        'party_id': self.party_id,
                        'roles': job_data.get("f_roles"),
                        'model_id': self.model_id,
                        'model_version': self.model_version,
                        'initiator_role': job_data.get('f_initiator_role'),
                        'initiator_party_id': job_data.get('f_initiator_party_id'),
                        'runtime_conf': job_data.get('f_runtime_conf'),
                        'work_mode': job_data.get('f_work_mode'),
                        'train_dsl': job_data.get('f_dsl'),
                        'train_runtime_conf': job_data.get('f_train_runtime_conf'),
                        'size': self.get_model_size(),
                        'job_status': job_data.get('f_status'),

                        'parent': False if json_loads(pipeline.inference_dsl) else True,
                        'fate_version': pipeline.fate_version,
                        'runtime_conf_on_party': json_loads(pipeline.runtime_conf_on_party),
                        'parent_info': json_loads(pipeline.parent_info),
                        'inference_dsl': json_loads(pipeline.inference_dsl)
                    }
                    model_utils.save_model_info(model_info)

                    schedule_logger(self.job_id).info(
                        'save {} model info done. model id: {}, model version: {}.'.format(self.job_id,
                                                                                           self.model_id,
                                                                                           self.model_version))
                else:
                    schedule_logger(self.job_id).info(
                        'save {} model info failed, no job found in db. '
                        'model id: {}, model version: {}.'.format(self.job_id,
                                                                  self.model_id,
                                                                  self.model_version))
            else:
                schedule_logger(self.job_id).info('model {} info has already existed in database.'.format(self.job_id))
        except Exception as e:
            schedule_logger(self.job_id).exception(e)

    @classmethod
    def get_dynamic_db_model(cls, base, job_id):
        return type(base.model(table_index=cls.get_dynamic_tracking_table_index(job_id=job_id)))

    @classmethod
    def get_dynamic_tracking_table_index(cls, job_id):
        return job_id[:8]

    def get_model_size(self):
        return self.pipelined_model.calculate_model_file_size()
