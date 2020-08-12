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

from fate_arch.storage.constant import StorageTableMetaType
from fate_flow.manager.table_manager.table_convert import convert
from fate_flow.entity.metric import MetricMeta

from arch.api.utils import log_utils
from fate_flow.manager.table_manager.table_operation import get_table
from fate_flow.utils.job_utils import generate_session_id

LOGGER = log_utils.getLogger()


class Reader(object):
    def __init__(self):
        self.data_output = None
        self.task_id = ''
        self.tracker = None
        self.parameters = None

    def run(self, component_parameters=None, args=None):
        self.parameters = component_parameters["ReaderParam"]
        job_id = generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id)
        table_key = [key for key in self.parameters.keys()][0]
        data_table = get_table(job_id=job_id,
                               namespace=self.parameters[table_key]['namespace'],
                               name=self.parameters[table_key]['name']
                               )
        if not data_table:
            raise Exception('no find table: namespace {}, name {}'.format(self.parameters[table_key]['namespace'],
                                                                          self.parameters[table_key]['name']))
        persistent_table_namespace, persistent_table_name = 'output_data_{}'.format(self.task_id), uuid.uuid1().hex
        table = convert(data_table, job_id=generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id),
                        name=persistent_table_name, namespace=persistent_table_namespace, force=True, mode=component_parameters['job_parameters']['work_mode'])
        if not table:
            persistent_table_name = data_table.get_name()
            persistent_table_namespace = data_table.get_namespace()
        partitions = data_table.get_partitions()
        count = data_table.count()
        LOGGER.info('save data view:name {}, namespace {}, partitions {}, count {}'.format(persistent_table_name,
                                                                                           persistent_table_namespace,
                                                                                           partitions,
                                                                                           count))
        self.tracker.log_output_data_info(data_name=component_parameters.get('output_data_name')[0] if component_parameters.get('output_data_name') else table_key,
                                          table_namespace=persistent_table_namespace,
                                          table_name=persistent_table_name)
        headers_str = data_table.get_meta(_type=StorageTableMetaType.SCHEMA).get('header')
        data_list = [headers_str.split(',')]
        party_of_data = data_table.get_meta(_type=StorageTableMetaType.PART_OF_DATA)
        for data in party_of_data:
            data_list.append(data[1].split(','))
        data = np.array(data_list)
        Tdata = data.transpose()
        table_info = {}
        for data in Tdata:
            table_info[data[0]] = ','.join(list(set(data[1:]))[:5])
        data_info = {
            "table_name": self.parameters[table_key]['name'],
            "table_info": table_info
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
