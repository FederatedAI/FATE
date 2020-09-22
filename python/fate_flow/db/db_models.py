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
import datetime
import inspect
import os
import sys

import __main__
from peewee import (CharField, IntegerField, BigIntegerField,
                    TextField, CompositeKey, BigAutoField, BooleanField)
from playhouse.apsw_ext import APSWDatabase
from playhouse.pool import PooledMySQLDatabase

from fate_arch.common import log
from fate_arch.common.base_utils import current_timestamp
from fate_arch.storage.metastore.base_model import JSONField, BaseModel, LongTextField
from fate_arch.common import WorkMode
from fate_flow.settings import DATABASE, WORK_MODE, stat_logger
from fate_flow.entity.runtime_config import RuntimeConfig


LOGGER = log.getLogger()


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


@singleton
class BaseDataBase(object):
    def __init__(self):
        database_config = DATABASE.copy()
        db_name = database_config.pop("name")
        if WORK_MODE == WorkMode.STANDALONE:
            self.database_connection = APSWDatabase('fate_flow_sqlite.db')
            RuntimeConfig.init_config(USE_LOCAL_DATABASE=True)
            stat_logger.info('init sqlite database on standalone mode successfully')
        elif WORK_MODE == WorkMode.CLUSTER:
            self.database_connection = PooledMySQLDatabase(db_name, **database_config)
            stat_logger.info('init mysql database on cluster mode successfully')
            RuntimeConfig.init_config(USE_LOCAL_DATABASE=False)
        else:
            raise Exception('can not init database')


MAIN_FILE_PATH = os.path.realpath(__main__.__file__)
if MAIN_FILE_PATH.endswith('fate_flow_server.py') or \
        MAIN_FILE_PATH.endswith('task_executor.py') or \
        MAIN_FILE_PATH.find("/unittest/__main__.py"):
    DB = BaseDataBase().database_connection
else:
    # Initialize the database only when the server is started.
    DB = None


def close_connection():
    try:
        if DB:
            DB.close()
    except Exception as e:
        LOGGER.exception(e)


class DataBaseModel(BaseModel):
    class Meta:
        database = DB


@DB.connection_context()
def init_database_tables():
    members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    table_objs = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):
            table_objs.append(obj)
    DB.create_tables(table_objs)


def fill_db_model_object(model_object, human_model_dict):
    for k, v in human_model_dict.items():
        attr_name = 'f_%s' % k
        if hasattr(model_object.__class__, attr_name):
            setattr(model_object, attr_name, v)
    return model_object


class Job(DataBaseModel):
    # multi-party common configuration
    f_job_id = CharField(max_length=25)
    f_name = CharField(max_length=500, null=True, default='')
    f_description = TextField(null=True, default='')
    f_tag = CharField(max_length=50, null=True, index=True, default='')
    f_dsl = JSONField()
    f_runtime_conf = JSONField()
    f_train_runtime_conf = JSONField(null=True)
    f_roles = JSONField()
    f_work_mode = IntegerField()
    f_initiator_role = CharField(max_length=50, index=True)
    f_initiator_party_id = CharField(max_length=50, index=True, default=-1)
    f_status = CharField(max_length=50)
    # this party configuration
    f_role = CharField(max_length=50, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_is_initiator = BooleanField(null=True, index=True, default=False)
    f_progress = IntegerField(null=True, default=0)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)
    f_start_time = BigIntegerField(null=True)
    f_end_time = BigIntegerField(null=True)
    f_elapsed = BigIntegerField(null=True)

    class Meta:
        db_table = "t_job"
        primary_key = CompositeKey('f_job_id', 'f_role', 'f_party_id')


class Task(DataBaseModel):
    # multi-party common configuration
    f_job_id = CharField(max_length=25)
    f_component_name = TextField()
    f_task_id = CharField(max_length=100)
    f_task_version = BigIntegerField()
    f_initiator_role = CharField(max_length=50, index=True)
    f_initiator_party_id = CharField(max_length=50, index=True, default=-1)
    f_federated_mode = CharField(max_length=10, index=True)
    f_federated_status_collect_type = CharField(max_length=10, index=True)
    f_status = CharField(max_length=50)
    # this party configuration
    f_role = CharField(max_length=50, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_run_on = BooleanField(null=True, index=True, default=False)
    f_run_ip = CharField(max_length=100, null=True)
    f_run_pid = IntegerField(null=True)
    f_party_status = CharField(max_length=50)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)
    f_start_time = BigIntegerField(null=True)
    f_end_time = BigIntegerField(null=True)
    f_elapsed = BigIntegerField(null=True)

    class Meta:
        db_table = "t_task"
        primary_key = CompositeKey('f_job_id', 'f_task_id', 'f_task_version', 'f_role', 'f_party_id')


class TrackingMetric(DataBaseModel):
    _mapper = {}

    @classmethod
    def model(cls, table_index=None, date=None):
        if not table_index:
            table_index = date.strftime(
                '%Y%m%d') if date else datetime.datetime.now().strftime(
                '%Y%m%d')
        class_name = 'TrackingMetric_%s' % table_index

        ModelClass = TrackingMetric._mapper.get(class_name, None)
        if ModelClass is None:
            class Meta:
                db_table = '%s_%s' % ('t_tracking_metric', table_index)

            attrs = {'__module__': cls.__module__, 'Meta': Meta}
            ModelClass = type("%s_%s" % (cls.__name__, table_index), (cls,),
                              attrs)
            TrackingMetric._mapper[class_name] = ModelClass
        return ModelClass()

    f_id = BigAutoField(primary_key=True)
    f_job_id = CharField(max_length=25)
    f_component_name = TextField()
    f_task_id = CharField(max_length=100, null=True)
    f_task_version = BigIntegerField(null=True)
    f_role = CharField(max_length=50, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_metric_namespace = CharField(max_length=180, index=True)
    f_metric_name = CharField(max_length=180, index=True)
    f_key = CharField(max_length=200)
    f_value = TextField()
    f_type = IntegerField(index=True)  # 0 is data, 1 is meta
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)


class TrackingOutputDataInfo(DataBaseModel):
    _mapper = {}

    @classmethod
    def model(cls, table_index=None, date=None):
        if not table_index:
            table_index = date.strftime(
                '%Y%m%d') if date else datetime.datetime.now().strftime(
                '%Y%m%d')
        class_name = 'TrackingOutputDataInfo_%s' % table_index

        ModelClass = TrackingOutputDataInfo._mapper.get(class_name, None)
        if ModelClass is None:
            class Meta:
                db_table = '%s_%s' % ('t_tracking_output_data_info', table_index)
                primary_key = CompositeKey('f_job_id', 'f_task_id', 'f_task_version', 'f_data_name', 'f_role', 'f_party_id')

            attrs = {'__module__': cls.__module__, 'Meta': Meta}
            ModelClass = type("%s_%s" % (cls.__name__, table_index), (cls,),
                              attrs)
            TrackingOutputDataInfo._mapper[class_name] = ModelClass
        return ModelClass()

    # multi-party common configuration
    f_job_id = CharField(max_length=25)
    f_component_name = TextField()
    f_task_id = CharField(max_length=100, null=True)
    f_task_version = BigIntegerField(null=True)
    f_data_name = CharField(max_length=30)
    # this party configuration
    f_role = CharField(max_length=50, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_table_name = CharField(max_length=500, null=True)
    f_table_namespace = CharField(max_length=500, null=True)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)
    f_description = TextField(null=True, default='')


class MachineLearningModelInfo(DataBaseModel):
    f_id = BigAutoField(primary_key=True)
    f_role = CharField(max_length=50, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_roles = JSONField()
    f_job_id = CharField(max_length=25)
    f_model_id = CharField(max_length=100, index=True)
    f_model_version = CharField(max_length=100, index=True)
    f_loaded_times = IntegerField(default=0)
    f_size = BigIntegerField(default=0)
    f_create_time = BigIntegerField(default=0)
    f_update_time = BigIntegerField(default=0)
    f_description = TextField(null=True, default='')
    f_initiator_role = CharField(max_length=50, index=True)
    f_initiator_party_id = CharField(max_length=50, index=True, default=-1)
    f_runtime_conf = JSONField()
    f_work_mode = IntegerField()
    f_dsl = JSONField()
    f_train_runtime_conf = JSONField(default={})
    f_imported = IntegerField(default=0)
    f_job_status = CharField(max_length=50)

    class Meta:
        db_table = "t_machine_learning_model_info"


class ModelTag(DataBaseModel):
    f_id = BigAutoField(primary_key=True)
    f_m_id = BigIntegerField(null=False)
    f_t_id = BigIntegerField(null=False)

    class Meta:
        db_table = "t_model_tag"


class Tag(DataBaseModel):
    f_id = BigAutoField(primary_key=True)
    f_name = CharField(max_length=100, index=True, unique=True)
    f_desc = TextField(null=True)
    f_create_time = BigIntegerField(default=current_timestamp())
    f_update_time = BigIntegerField(default=current_timestamp())

    class Meta:
        db_table = "t_tags"


class ComponentSummary(DataBaseModel):
    _mapper = {}

    @classmethod
    def model(cls, table_index=None, date=None):
        if not table_index:
            table_index = date.strftime(
                '%Y%m%d') if date else datetime.datetime.now().strftime(
                '%Y%m%d')
        class_name = 'ComponentSummary_%s' % table_index

        ModelClass = TrackingMetric._mapper.get(class_name, None)
        if ModelClass is None:
            class Meta:
                db_table = '%s_%s' % ('t_component_summary', table_index)

            attrs = {'__module__': cls.__module__, 'Meta': Meta}
            ModelClass = type("%s_%s" % (cls.__name__, table_index), (cls,), attrs)
            ComponentSummary._mapper[class_name] = ModelClass
        return ModelClass()

    f_id = BigAutoField(primary_key=True)
    f_job_id = CharField(max_length=25)
    f_role = CharField(max_length=25, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_component_name = TextField()
    f_task_id = CharField(max_length=50, null=True)
    f_task_version = CharField(max_length=50, null=True)
    f_summary = LongTextField()
    f_create_time = BigIntegerField(default=0)
    f_update_time = BigIntegerField(default=0)


class ModelOperationLog(DataBaseModel):
    f_operation_type = CharField(max_length=20, null=False, index=True)
    f_operation_status = CharField(max_length=20, null=True, index=True)
    f_initiator_role = CharField(max_length=50, index=True, null=True)
    f_initiator_party_id = CharField(max_length=10, index=True, null=True)
    f_request_ip = CharField(max_length=20, null=True)
    f_model_id = CharField(max_length=100, index=True)
    f_model_version = CharField(max_length=100, index=True)
    f_create_time = BigIntegerField(default=current_timestamp())
    f_update_time = BigIntegerField(default=current_timestamp())

    class Meta:
        db_table = "t_model_operation_log"


class BackendRegistry(DataBaseModel):
    f_engine_name = CharField(max_length=50, index=True)
    f_engine_type = CharField(max_length=10, index=True)
    f_engine_address = JSONField()
    f_cores = IntegerField(index=True)
    f_memory = IntegerField(index=True)  # MB
    f_remaining_cores = IntegerField(index=True)
    f_remaining_memory = IntegerField(index=True) # MB
    f_nodes = IntegerField(index=True)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)

    class Meta:
        db_table = "t_backend_registry"
        primary_key = CompositeKey('f_engine_name', 'f_engine_type')


class ResourceRecord(DataBaseModel):
    f_job_id = CharField(max_length=25)
    f_role = CharField(max_length=50, index=True)
    f_party_id = CharField(max_length=10, index=True)
    f_engine_name = CharField(max_length=50, index=True)
    f_engine_type = CharField(max_length=10, index=True)
    f_cores = IntegerField(index=True)
    f_memory = IntegerField(index=True)  # MB
    f_remaining_cores = IntegerField(index=True, default=0)
    f_remaining_memory = IntegerField(index=True, default=0)  # MB
    f_in_use = BooleanField(index=True, default=True)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)

    class Meta:
        db_table = "t_resource_record"
        primary_key = CompositeKey('f_job_id', 'f_role', 'f_party_id')


class DBQueue(DataBaseModel):
    f_job_id = CharField(max_length=25, primary_key=True)
    f_job_status = CharField(max_length=50, index=True)
    f_initiator_role = CharField(max_length=50, index=True)
    f_initiator_party_id = CharField(max_length=50, index=True, default=-1)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)
    f_tag = CharField(max_length=50, null=True, index=True, default='')

    class Meta:
        db_table = "t_queue"
