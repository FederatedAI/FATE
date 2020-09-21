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
import numpy as np

from fate_arch import storage
from fate_arch.abc import StorageTableABC, StorageTableMetaABC, AddressABC
from fate_arch.common import log, EngineType
from fate_arch.computing import ComputingEngine
from fate_arch.storage import StorageTableMeta, StorageEngine, Relationship
from fate_flow.entity.metric import MetricMeta
from fate_flow.utils import job_utils, data_utils

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
        computing_engine = component_parameters["job_parameters"]["computing_engine"]
        output_table_namespace, output_table_name = data_utils.default_output_table_info(task_id=self.task_id,
                                                                                         task_version=self.tracker.task_version)
        input_table_meta, output_table_address, output_table_engine = self.convert_check(
            input_name=self.parameters[table_key]['name'],
            input_namespace=self.parameters[table_key]['namespace'],
            output_name=output_table_name,
            output_namespace=output_table_namespace,
            computing_engine=computing_engine,
            output_storage_address=output_storage_address)
        with storage.Session.build(
                session_id=job_utils.generate_session_id(self.tracker.task_id, self.tracker.task_version,
                                                         self.tracker.role, self.tracker.party_id,
                                                         suffix="storage", random_end=True),
                storage_engine=input_table_meta.get_engine()) as input_table_session:
            input_table = input_table_session.get_table(name=input_table_meta.get_name(),
                                                        namespace=input_table_meta.get_namespace())
            # update real count to meta info
            input_table.count()
            if output_table_address:
                # Table replication is required
                if input_table_meta.get_engine() != output_table_engine:
                    LOGGER.info(
                        f"the {input_table_meta.get_engine()} engine input table needs to be converted to {output_table_engine} engine to support computing engine {computing_engine}")
                else:
                    LOGGER.info(f"the {input_table_meta.get_engine()} input table needs to be transform format")
                with storage.Session.build(
                        session_id=job_utils.generate_session_id(self.tracker.task_id, self.tracker.task_version,
                                                                 self.tracker.role, self.tracker.party_id,
                                                                 suffix="storage",
                                                                 random_end=True),
                        storage_engine=output_table_engine) as output_table_session:
                    output_table = output_table_session.create_table(address=output_table_address,
                                                                     name=output_table_name,
                                                                     namespace=output_table_namespace,
                                                                     partitions=input_table_meta.partitions)
                    self.copy_table(src_table=input_table, dest_table=output_table)
                    # update real count to meta info
                    output_table.count()
                    output_table_meta = output_table.get_meta()
            else:
                LOGGER.info(
                    f"the {input_table_meta.get_engine()} engine input table does not require conversion to support computing engine {computing_engine}")
                output_table_meta = input_table_meta
        self.tracker.log_output_data_info(
            data_name=component_parameters.get('output_data_name')[0] if component_parameters.get(
                'output_data_name') else table_key,
            table_namespace=output_table_meta.get_namespace(),
            table_name=output_table_meta.get_name())
        headers_str = output_table_meta.get_schema().get('header')
        table_info = {}
        if output_table_meta.get_schema() and headers_str:
            data_list = [headers_str.split(',')]
            for data in output_table_meta.get_part_of_data():
                data_list.append(data[1].split(','))
            data = np.array(data_list)
            Tdata = data.transpose()
            for data in Tdata:
                table_info[data[0]] = ','.join(list(set(data[1:]))[:5])
        data_info = {
            "table_name": self.parameters[table_key]['name'],
            "namespace": self.parameters[table_key]['namespace'],
            "table_info": table_info,
            "partitions": output_table_meta.get_partitions(),
            "storage_engine": output_table_meta.get_engine(),
            "count": output_table_meta.get_count()
        }

        self.tracker.set_metric_meta(metric_namespace="reader_namespace",
                                     metric_name="reader_name",
                                     metric_meta=MetricMeta(name='reader', metric_type='data_info',
                                                            extra_metas=data_info))

    def convert_check(self, input_name, input_namespace, output_name, output_namespace,
                      computing_engine: ComputingEngine = ComputingEngine.EGGROLL, output_storage_address={}) -> (
            StorageTableMetaABC, AddressABC, StorageEngine):
        input_table_meta = StorageTableMeta(name=input_name, namespace=input_namespace)
        if not input_table_meta:
            raise RuntimeError(f"can not found table name: {input_name} namespace: {input_namespace}")
        output_table_address = None
        output_table_engine = None
        if input_table_meta.get_engine() not in Relationship.CompToStore.get(computing_engine,
                                                                             []) or not input_table_meta.get_in_serialized():
            address_dict = output_storage_address.copy()
            if computing_engine == ComputingEngine.STANDALONE:
                from fate_arch.storage import StandaloneStorageType
                address_dict["name"] = output_name
                address_dict["namespace"] = output_namespace
                address_dict["storage_type"] = StandaloneStorageType.ROLLPAIR_LMDB
                output_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.STANDALONE,
                                                                       address_dict=address_dict)
                output_table_engine = StorageEngine.STANDALONE
            elif computing_engine == ComputingEngine.EGGROLL:
                from fate_arch.storage import EggRollStorageType
                address_dict["name"] = output_name
                address_dict["namespace"] = output_namespace
                address_dict["storage_type"] = EggRollStorageType.ROLLPAIR_LMDB
                output_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.EGGROLL,
                                                                       address_dict=address_dict)
                output_table_engine = StorageEngine.EGGROLL
            elif computing_engine == ComputingEngine.SPARK:
                address_dict["path"] = data_utils.default_output_fs_path(name=output_name, namespace=output_namespace, prefix=address_dict.get("path_prefix"))
                output_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.HDFS,
                                                                       address_dict=address_dict)
                output_table_engine = StorageEngine.HDFS
            else:
                raise RuntimeError(f"can not support computing engine {computing_engine}")
            return input_table_meta, output_table_address, output_table_engine
        else:
            return input_table_meta, output_table_address, output_table_engine

    def copy_table(self, src_table: StorageTableABC, dest_table: StorageTableABC):
        count = 0
        data_temp = []
        part_of_data = []
        src_table_meta = src_table.get_meta()
        LOGGER.info(f"start copying table")
        LOGGER.info(
            f"source table name: {src_table.get_name()} namespace: {src_table.get_namespace()} engine: {src_table.get_engine()}")
        LOGGER.info(
            f"destination table name: {dest_table.get_name()} namespace: {dest_table.get_namespace()} engine: {dest_table.get_engine()}")
        schema = {}
        if not src_table_meta.get_in_serialized():
            if src_table_meta.get_have_head():
                get_head = False
            else:
                get_head = True
            for line in src_table.read():
                if not get_head:
                    schema = data_utils.get_header_schema(header_line=line, id_delimiter=src_table_meta.get_id_delimiter())
                    get_head = True
                    continue
                values = line.rstrip().split(src_table.get_meta().get_id_delimiter())
                k, v = values[0], data_utils.list_to_str(values[1:],
                                                         id_delimiter=src_table.get_meta().get_id_delimiter())
                count = self.put_in_table(table=dest_table, k=k, v=v, temp=data_temp, count=count,
                                          part_of_data=part_of_data)
        else:
            for k, v in src_table.collect():
                count = self.put_in_table(table=dest_table, k=k, v=v, temp=data_temp, count=count,
                                          part_of_data=part_of_data)
            schema = src_table.get_meta().get_schema()
        if data_temp:
            dest_table.put_all(data_temp)
        LOGGER.info("copy successfully")
        dest_table.get_meta().update_metas(schema=schema, part_of_data=part_of_data)

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
