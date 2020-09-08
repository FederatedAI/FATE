#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import numpy as np

from fate_arch.computing import ComputingEngine
from fate_flow.entity.metric import MetricMeta
from fate_arch.common import log
from fate_arch.storage import StorageTableMeta, StorageEngine, Relationship
from fate_arch import storage
from fate_flow.utils import job_utils
from fate_arch.abc import StorageTableABC, StorageTableMetaABC, AddressABC

LOGGER = log.getLogger()
MAX_NUM = 10000


class Reader(object):
    def __init__(self):
        self.data_output = None
        self.task_id = ''
        self.tracker = None
        self.parameters = None

    def run(self, component_parameters=None, args=None):
        self.parameters = component_parameters["ReaderParam"]
        table_key = [key for key in self.parameters.keys()][0]
        persistent_table_namespace, persistent_table_name = 'output_data_{}'.format(self.task_id), uuid.uuid1().hex
        src_table_meta, dest_table_address, dest_table_engine = self.convert(src_name=self.parameters[table_key]['name'],
                                                                             src_namespace=self.parameters[table_key]['namespace'],
                                                                             dest_name=persistent_table_name,
                                                                             dest_namespace=persistent_table_namespace,
                                                                             computing_engine=component_parameters.get('job_parameters').get('computing_engine', ComputingEngine.EGGROLL),
                                                                             force=True)
        if dest_table_address:
            with storage.Session.build(session_id=job_utils.generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id, suffix="storage", random_end=True),
                                       storage_engine=dest_table_engine) as storage_session:
                dest_table = storage_session.create_table(address=dest_table_address, name=persistent_table_name, namespace=persistent_table_namespace, partitions=src_table_meta.partitions)
                src_table = storage_session.get_table(name=src_table_meta.get_name(), namespace=src_table_meta.get_namespace())
                self.copy_table(src_table=src_table, dest_table=dest_table)
                dest_table.count()
                dest_table_meta = dest_table.get_meta()
        else:
            dest_table_meta = src_table_meta
        self.tracker.log_output_data_info(data_name=component_parameters.get('output_data_name')[0] if component_parameters.get('output_data_name') else table_key,
                                          table_namespace=dest_table_meta.get_namespace(),
                                          table_name=dest_table_meta.get_name())
        headers_str = dest_table_meta.get_schema().get('header')
        table_info = {}
        if dest_table_meta.get_schema() and headers_str:
            data_list = [headers_str.split(',')]
            for data in dest_table_meta.get_part_of_data():
                data_list.append(data[1].split(','))
            data = np.array(data_list)
            Tdata = data.transpose()
            for data in Tdata:
                table_info[data[0]] = ','.join(list(set(data[1:]))[:5])
        data_info = {
            "table_name": self.parameters[table_key]['name'],
            "namespace": self.parameters[table_key]['namespace'],
            "table_info": table_info,
            "partitions": dest_table_meta.get_partitions(),
            "storage_engine": dest_table_meta.get_engine(),
            "count": dest_table_meta.get_count()
        }

        self.tracker.set_metric_meta(metric_namespace="reader_namespace",
                                     metric_name="reader_name",
                                     metric_meta=MetricMeta(name='reader', metric_type='data_info', extra_metas=data_info))

    def convert(self, src_name, src_namespace, dest_name, dest_namespace,
                computing_engine: ComputingEngine = ComputingEngine.EGGROLL, force=False) -> (StorageTableMetaABC, AddressABC, StorageEngine):
        # The source and target may be different session types
        src_table_meta = StorageTableMeta(name=src_name, namespace=src_namespace)
        if not src_table_meta:
            raise RuntimeError(f"can not found table name: {src_name} namespace: {src_namespace}")
        dest_table_address = None
        dest_table_engine = None
        if src_table_meta.get_engine() not in Relationship.CompToStore.get(computing_engine, []):
            if computing_engine == ComputingEngine.STANDALONE:
                from fate_arch.storage import EggRollStorageType
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.STANDALONE,
                                                                     address_dict=dict(name=dest_name,
                                                                                       namespace=dest_namespace,
                                                                                       storage_type=EggRollStorageType.ROLLPAIR_LMDB))
                dest_table_engine = StorageEngine.STANDALONE
            elif computing_engine == ComputingEngine.EGGROLL:
                from fate_arch.storage import EggRollStorageType
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.EGGROLL,
                                                                     address_dict=dict(name=dest_name,
                                                                                       namespace=dest_namespace,
                                                                                       storage_type=EggRollStorageType.ROLLPAIR_LMDB))
                dest_table_engine = StorageEngine.EGGROLL
            elif computing_engine == ComputingEngine.SPARK:
                pass
            else:
                raise RuntimeError(f"can not support computing engine {computing_engine}")
            return src_table_meta, dest_table_address, dest_table_engine
        elif src_table_meta.get_engine() == StorageEngine.HDFS:
            dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.HDFS,
                                                                 address_dict=dict(path=f"{src_table_meta.get_address().path}_fate_{self.task_id}"))
            dest_table_engine = StorageEngine.HDFS
            return src_table_meta, dest_table_address, dest_table_engine
        else:
            return src_table_meta, dest_table_address, dest_table_engine

    def copy_table(self, src_table: StorageTableABC, dest_table: StorageTableABC):
        count = 0
        data = []
        part_of_data = []
        for k, v in src_table.collect():
            data.append((k, v))
            count += 1
            if count < 100:
                part_of_data.append((k, v))
            if len(data) == MAX_NUM:
                dest_table.put_all(data)
                data = []
        if data:
            dest_table.put_all(data)
        dest_table.get_meta().update_metas(schema=src_table.get_meta().get_schema(), count=src_table.count(),
                                           part_of_data=part_of_data)

    def set_taskid(self, taskid):
        self.task_id = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None
