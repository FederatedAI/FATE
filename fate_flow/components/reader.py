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
from fate_arch.common import log, EngineType
from fate_arch.storage import StorageTableMeta, StorageEngine, Relationship, DEFAULT_ID_DELIMITER
from fate_arch import storage
from fate_flow.utils import job_utils, data_utils
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
        output_storage_address = args["job_parameters"].engines_address[EngineType.STORAGE]
        table_key = [key for key in self.parameters.keys()][0]
        output_table_namespace, output_table_name = data_utils.default_output_table_info(task_id=self.task_id, task_version=self.tracker.task_version)
        src_table_meta, dest_table_address, dest_table_engine = self.convert_check(src_name=self.parameters[table_key]['name'],
                                                                                   src_namespace=self.parameters[table_key]['namespace'],
                                                                                   dest_name=output_table_name,
                                                                                   dest_namespace=output_table_namespace,
                                                                                   computing_engine=component_parameters.get('job_parameters').get('computing_engine', ComputingEngine.EGGROLL),
                                                                                   output_storage_address=output_storage_address)
        if dest_table_address:
            # Table replication is required
            with storage.Session.build(session_id=job_utils.generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id, suffix="storage", random_end=True),
                                       storage_engine=dest_table_engine) as dest_storage_session:
                dest_table = dest_storage_session.create_table(address=dest_table_address, name=output_table_name, namespace=output_table_namespace, partitions=src_table_meta.partitions)
                with storage.Session.build(session_id=job_utils.generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id, suffix="storage", random_end=True),
                                           storage_engine=src_table_meta.get_engine()) as src_storage_session:
                    src_table = src_storage_session.get_table(name=src_table_meta.get_name(), namespace=src_table_meta.get_namespace())
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

    def convert_check(self, src_name, src_namespace, dest_name, dest_namespace,
                      computing_engine: ComputingEngine = ComputingEngine.EGGROLL, output_storage_address={}) -> (StorageTableMetaABC, AddressABC, StorageEngine):
        src_table_meta = StorageTableMeta(name=src_name, namespace=src_namespace)
        if not src_table_meta:
            raise RuntimeError(f"can not found table name: {src_name} namespace: {src_namespace}")
        dest_table_address = None
        dest_table_engine = None
        if src_table_meta.get_engine() not in Relationship.CompToStore.get(computing_engine, []) or not src_table_meta.get_in_serialized():
            address_dict = output_storage_address.copy()
            if computing_engine == ComputingEngine.STANDALONE:
                from fate_arch.storage import StandaloneStorageType
                address_dict["name"] = dest_name
                address_dict["namespace"] = dest_namespace
                address_dict["storage_type"] = StandaloneStorageType.ROLLPAIR_LMDB
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.STANDALONE,
                                                                     address_dict=address_dict)
                dest_table_engine = StorageEngine.STANDALONE
            elif computing_engine == ComputingEngine.EGGROLL:
                from fate_arch.storage import EggRollStorageType
                address_dict["name"] = dest_name
                address_dict["namespace"] = dest_namespace
                address_dict["storage_type"] = EggRollStorageType.ROLLPAIR_LMDB
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.EGGROLL,
                                                                     address_dict=address_dict)
                dest_table_engine = StorageEngine.EGGROLL
            elif computing_engine == ComputingEngine.SPARK:
                address_dict["path"] = data_utils.default_output_path(name=dest_name, namespace=dest_namespace)
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.HDFS,
                                                                     address_dict=address_dict)
                dest_table_engine = StorageEngine.HDFS
            else:
                raise RuntimeError(f"can not support computing engine {computing_engine}")
            return src_table_meta, dest_table_address, dest_table_engine
        else:
            return src_table_meta, dest_table_address, dest_table_engine

    def copy_table(self, src_table: StorageTableABC, dest_table: StorageTableABC):
        count = 0
        data_temp = []
        part_of_data = []
        src_table_meta = src_table.get_meta()
        if not src_table_meta.get_in_serialized():
            if src_table_meta.get_have_head():
                get_head = False
            else:
                get_head = True
            for line in src_table.read():
                if not get_head:
                    dest_table.get_meta().update_metas(schema=data_utils.get_header_schema(header_line=line, id_delimiter=src_table_meta.get_id_delimiter()))
                    get_head = True
                    continue
                values = line.rstrip().split(self.parameters["id_delimiter"])
                k, v = values[0], data_utils.list_to_str(values[1:], id_delimiter=self.parameters["id_delimiter"])
                count = self.put_in_table(table=dest_table, k=k, v=v, temp=data_temp, count=count, part_of_data=part_of_data)
        else:
            for k, v in src_table.collect():
                count = self.put_in_table(table=dest_table, k=k, v=v, temp=data_temp, count=count, part_of_data=part_of_data)
        if data_temp:
            dest_table.put_all(data_temp)
        dest_table.get_meta().update_metas(schema=src_table.get_meta().get_schema(), count=count,
                                           part_of_data=part_of_data)

    def put_in_table(self, table: StorageTableABC, k, v, temp, count, part_of_data):
        temp.append((k, v))
        if count < 100:
            part_of_data.append((k, v))
        if len(temp) == MAX_NUM:
            table.put_all(temp)
            temp.clear()
        return count + 1

    def set_taskid(self, taskid):
        self.task_id = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None
