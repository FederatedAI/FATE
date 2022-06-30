#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : upload_db.py
# @Author: Richard Chiming Xu
# @Date  : 2022/5/13
# @Desc  :

import time
import pandas as pd

from federatedml.param.upload_from_db_param import UploadFromDBParam
from fate_arch.common import log, EngineType
from fate_arch.storage import StorageEngine, EggRollStoreType
from fate_flow.entity import Metric, MetricMeta
from federatedml.upload_from_db import db_utils
from fate_flow.utils import job_utils, data_utils
from fate_flow.scheduling_apps.client import ControllerClient
from fate_arch.common.data_utils import default_input_fs_path
from fate_arch import storage, session
from fate_arch.session import Session
from federatedml.model_base import ModelBase

LOGGER = log.getLogger()


class UploadFromDB(ModelBase):
    def __init__(self):
        super(UploadFromDB, self).__init__()
        self.model_name = 'UploadFromDB'
        self.model_param_name = 'UploadFromDBParam'
        self.model_meta_name = 'UploadFromDBMeta'
        self.MAX_PARTITIONS = 1024
        self.MAX_BYTES = 1024 * 1024 * 8
        self.parameters = {}
        self.table = None
        self.session = None
        self.session_id = None
        self.storage_engine = None
        self.model_param = UploadFromDBParam()

    def _init_model(self, model):
        self.model_param = model

    def _run(self, cpn_input):

        LOGGER.info('begin to execute upload from db.')
        LOGGER.info('full domain: {}'.format(cpn_input))
        LOGGER.info('tracker: {}'.format(cpn_input.tracker))
        LOGGER.info('task_version_id: {}'.format(cpn_input.task_version_id))
        LOGGER.info('checkpoint_manager: {}'.format(cpn_input.checkpoint_manager))
        LOGGER.info('parameters: {}'.format(cpn_input.parameters))
        LOGGER.info('flow_feeded_parameters: {}'.format(cpn_input.flow_feeded_parameters))
        LOGGER.info('roles: {}'.format(cpn_input.roles))
        LOGGER.info('job_parameters: {}'.format(cpn_input.job_parameters))
        LOGGER.info('datasets: {}'.format(cpn_input.datasets))
        LOGGER.info('models: {}'.format(cpn_input.models))
        LOGGER.info('caches: {}'.format(cpn_input.caches))

        self.model_param.update(cpn_input.parameters)
        self.parameters = cpn_input.parameters
        self.tracker = cpn_input.tracker
        self.parameters["role"] = cpn_input.roles['role']
        self.parameters["local"] = cpn_input.roles['local']
        storage_engine = self.parameters["storage_engine"]
        storage_address = self.parameters["storage_address"]
        # if not set storage, use job storage as default
        if not storage_engine:
            storage_engine = cpn_input.job_parameters.storage_engine
        if not storage_address:
            storage_address = cpn_input.job_parameters.engines_address[EngineType.STORAGE]
        job_id = self.task_version_id.split("_")[0]

        # if not storage table_name namespace, use table_name db
        name, namespace = self.parameters.get("name"), self.parameters.get("namespace")
        # if namespace is None:
        #     namespace = component_parameters["table_name"]
        # if name is None:
        #     name = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if namespace is None:
            namespace = self.parameters["db"]
        if name is None:
            name = self.parameters["table_name"]

        partitions = self.parameters["partition"]
        if partitions <= 0 or partitions >= self.MAX_PARTITIONS:
            raise Exception("Error number of partition, it should between %d and %d" % (0, self.MAX_PARTITIONS))

        sess = Session.get_global()
        if self.parameters.get("destroy", False):
            table = sess.get_table(namespace=namespace, name=name)
            if table:
                LOGGER.info(
                    f"destroy table name: {name} namespace: {namespace} engine: {table.engine}"
                )
                try:
                    table.destroy()
                except Exception as e:
                    LOGGER.error(e)
            else:
                LOGGER.info(
                    f"can not found table name: {name} namespace: {namespace}, pass destroy"
                )
        address_dict = storage_address.copy()
        storage_session = sess.storage(
            storage_engine=storage_engine, options=self.parameters.get("options")
        )
        upload_address = {}
        if storage_engine in {StorageEngine.EGGROLL, StorageEngine.STANDALONE}:
            upload_address = {
                "name": name,
                "namespace": namespace,
                "storage_type": EggRollStoreType.ROLLPAIR_LMDB,
            }
        elif storage_engine in {StorageEngine.MYSQL, StorageEngine.HIVE}:
            if not address_dict.get("db") or not address_dict.get("name"):
                upload_address = {"db": namespace, "name": name}
        elif storage_engine in {StorageEngine.PATH}:
            upload_address = {"path": self.parameters["file"]}
        elif storage_engine in {StorageEngine.HDFS}:
            upload_address = {
                "path": default_input_fs_path(
                    name=name,
                    namespace=namespace,
                    prefix=address_dict.get("path_prefix"),
                )
            }
        elif storage_engine in {StorageEngine.LOCALFS}:
            upload_address = {
                "path": default_input_fs_path(
                    name=name,
                    namespace=namespace,
                    storage_engine=storage_engine
                )
            }
        else:
            raise RuntimeError(f"can not support this storage engine: {storage_engine}")

        address_dict.update(upload_address)
        LOGGER.info(f"upload to {storage_engine} storage, address: {address_dict}")

        address = storage.StorageTableMeta.create_address(storage_engine=storage_engine, address_dict=address_dict)
        self.parameters["partitions"] = partitions
        self.parameters["name"] = name
        self.table = storage_session.create_table(address=address, **self.parameters)
        # data_table_count = None
        data_table_count = self.save_data_table_from_db(job_id, name, namespace)
        self.table.get_meta().update_metas(in_serialized=True)
        LOGGER.info("------------load data finish!-----------------")
        LOGGER.info("total data_count: {}".format(data_table_count))
        LOGGER.info("table name: {}, table namespace: {}".format(name, namespace))

    def save_data_table_from_db(self, job_id, dst_table_name, dst_table_namespace):
        table_name = self.parameters['table']
        conn = db_utils.get_connection(host=self.parameters['host'],
                                       port=int(self.parameters['port']),
                                       user=self.parameters['user'],
                                       password=self.parameters['passwd'],
                                       db_name=self.parameters['db'],
                                       db_type=self.parameters['db_type'])
        cursor = conn.cursor()

        try:
            # upload display table meta
            table_count = self.get_table_record_count(cursor, table_name)
            self.update_meta()

            # query table records
            query_sql = "select {} from {}".format(','.join([self.parameters['id'], *self.parameters['params']]),
                                                   table_name)
            LOGGER.info("query sql is: {}".format(query_sql))
            cursor.execute(query_sql)
            batch_count = 1000
            parse_count = 0
            first_batch = True
            data = []

            result = cursor.fetchall()
            for r in result:
                data.append((r[0], data_utils.list_to_str(r[1:], id_delimiter=self.parameters["id_delimiter"])))
                if len(data) >= batch_count:
                    parse_count += len(data)
                    save_progress = parse_count / table_count * 100 // 1
                    job_info = {
                        'progress': save_progress,
                        "job_id": job_id,
                        "role": self.parameters["local"]['role'],
                        "party_id": self.parameters["local"]['party_id']
                    }
                    ControllerClient.update_job(job_info=job_info)
                    self.table.put_all(data)

                    # if this is the first batch, use first 100 records as example record
                    if first_batch:
                        first_batch = False
                        self.table.get_meta().update_metas(part_of_data=data[:100])

                    data = []

            if data:
                parse_count += len(data)
                self.table.put_all(data)
                if first_batch:
                    self.table.get_meta().update_metas(part_of_data=data[:100])
                table_count = self.table.count()
                self.table.get_meta().update_metas(count=table_count, partitions=self.parameters["partition"])
                self.save_meta(dst_table_namespace=dst_table_namespace, dst_table_name=dst_table_name,
                               table_count=table_count)

        finally:
            conn.close()
        return table_count

    def update_meta(self):
        header_schema = {
            'header': self.parameters["id_delimiter"].join(self.parameters['params']).strip(),
            'sid': self.parameters['id'].strip()
        }

        _, meta = self.table.meta.update_metas(
            schema=data_utils.get_header_schema(
                header_line=self.parameters["id_delimiter"].join(self.parameters['params']),
                id_delimiter=self.parameters["id_delimiter"],
                extend_sid=self.parameters['id'].strip(),
            ),
            auto_increasing_sid=False,
            extend_sid=self.parameters['id'].strip()
        )
        self.table.meta = meta

        # LOGGER.info("header schema:{}".format(header_schema))
        # self.table.get_meta().update_metas(header_schema)

    def save_meta(self, dst_table_namespace, dst_table_name, table_count):
        self.tracker.log_output_data_info(data_name='upload',
                                          table_namespace=dst_table_namespace,
                                          table_name=dst_table_name)
        self.tracker.log_metric_data(metric_namespace="upload",
                                     metric_name="data_access",
                                     metrics=[Metric("count", table_count)])
        self.tracker.set_metric_meta(metric_namespace="upload",
                                     metric_name="data_access",
                                     metric_meta=MetricMeta(name='upload', metric_type='UPLOAD'))

    @staticmethod
    def get_table_record_count(cursor, table_name):
        sql = "select count(*) from {}".format(table_name)
        LOGGER.info("read count sql: {}".format(sql))
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        LOGGER.info("total count:{}".format(count))
        return count
