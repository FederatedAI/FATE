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
from arch.api.data_table.table_convert import convert
from fate_flow.entity.metric import MetricMeta, Metric

from arch.api.utils import log_utils


LOGGER = log_utils.getLogger()


class Reader(object):
    def __init__(self):
        self.data_output = None
        self.task_id = ''
        self.tracker = None

    def run(self, component_parameters=None, args=None):
        data_table = args.get('data').get('args').get('data')[0]
        persistent_table_namespace, persistent_table_name = 'output_data_{}'.format(self.task_id), data_table.get_name()
        table = convert(data_table, name=persistent_table_name, namespace=persistent_table_namespace, force=True)
        self.tracker.save_data_view(
            data_info={'f_table_name':  persistent_table_name,
                       'f_table_namespace':  persistent_table_namespace,
                       'f_partition': table.get_partitions() if table else None,
                       'f_table_count_actual': table.count() if table else 0},
            mark=True)
        self.callback_metric(metric_name='reader_name',
                             metric_namespace='reader_namespace',
                             metric_data=[Metric("count", table.count()),
                                          Metric("partitions", table.get_partitions()),
                                          Metric("input_table_type", data_table.get_storage_engine()),
                                          Metric("output_table_type", table.get_storage_engine()),
                                          Metric("input_table_info", data_table.get_address()),
                                          Metric("output_table_info", table.get_storage_engine())
                                          ]
                             )

    def set_taskid(self, task_id):
        self.task_id = task_id

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None

    def callback_metric(self, metric_name, metric_namespace, metric_data):
        self.tracker.log_metric_data(metric_name=metric_name,
                                     metric_namespace=metric_namespace,
                                     metrics=metric_data)
        self.tracker.set_metric_meta(metric_namespace,
                                     metric_name,
                                     MetricMeta(name='reader',
                                                metric_type='READER'))

