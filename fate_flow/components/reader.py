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

from fate_flow.entity.metric import MetricMeta
from fate_arch.common import log
from fate_arch import storage
from fate_flow.utils.job_utils import generate_session_id

LOGGER = log.getLogger()


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
        src_table_meta, dest_table_address, dest_table_engine = storage.Session.convert(src_name=self.parameters[table_key]['name'],
                                                                                        src_namespace=self.parameters[table_key]['namespace'],
                                                                                        dest_name=persistent_table_name,
                                                                                        dest_namespace=persistent_table_namespace,
                                                                                        force=True)
        if dest_table_address:
            with storage.Session.build(session_id=generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id, suffix="storage"),
                                       storage_engine=dest_table_engine) as dest_session:
                dest_table = dest_session.create_table(address=dest_table_address, name=persistent_table_name, namespace=persistent_table_namespace, partitions=src_table_meta.partitions)
                dest_table.count()
                dest_table_meta = dest_table.get_meta()
        else:
            dest_table_meta = src_table_meta
        self.tracker.log_output_data_info(data_name=component_parameters.get('output_data_name')[0] if component_parameters.get('output_data_name') else table_key,
                                          table_namespace=dest_table_meta.get_namespace(),
                                          table_name=dest_table_meta.get_name())
        headers_str = dest_table_meta.get_schema().get('header')
        data_list = [headers_str.split(',')]
        for data in dest_table_meta.get_part_of_data():
            data_list.append(data[1].split(','))
        data = np.array(data_list)
        Tdata = data.transpose()
        table_info = {}
        for data in Tdata:
            table_info[data[0]] = ','.join(list(set(data[1:]))[:5])
        data_info = {
            "table_name": self.parameters[table_key]['name'],
            "table_info": table_info,
            "partitions": dest_table_meta.get_partitions(),
            "storage_engine": dest_table_meta.get_engine(),
            "count": dest_table_meta.get_count()
        }

        self.tracker.set_metric_meta(metric_namespace="reader_namespace",
                                     metric_name="reader_name",
                                     metric_meta=MetricMeta(name='reader', metric_type='data_info', extra_metas=data_info))

    def set_taskid(self, taskid):
        self.task_id = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None
