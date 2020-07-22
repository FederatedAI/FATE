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

from fate_arch.data_table.table_convert import convert
from fate_flow.entity.metric import MetricMeta

from arch.api.utils import log_utils
from fate_flow.utils.job_utils import generate_session_id

LOGGER = log_utils.getLogger()


class Reader(object):
    def __init__(self):
        self.data_output = None
        self.task_id = ''
        self.tracker = None

    def run(self, component_parameters=None, args=None):
        data_table = args.get('data').get('args').get('data')[0]
        persistent_table_namespace, persistent_table_name = 'output_data_{}'.format(self.task_id), uuid.uuid1().hex
        table = convert(data_table, job_id=generate_session_id(self.task_id, self.tracker.role, self.tracker.party_id),
                        name=persistent_table_name, namespace=persistent_table_namespace, force=True)
        self.tracker.save_data_view(
            data_info={'f_table_name':  persistent_table_name if table else data_table.get_name(),
                       'f_table_namespace':  persistent_table_namespace if table else data_table.get_namespace(),
                       'f_partition': table.get_partitions() if table else data_table.get_partitions(),
                       'f_table_count_actual': table.count() if table else data_table.get_partitions()},
            mark=True)
        self.callback_metric(metric_name='reader_name',
                             metric_namespace='reader_namespace',
                             data_info={"count": table.count(),
                                        "partitions": table.get_partitions(),
                                        "input_table_strage_engine": data_table.get_storage_engine(),
                                        "output_table_strage_engine": table.get_storage_engine()}
                             )
        data_table.close()
        table.close()

    def set_taskid(self, task_id):
        self.task_id = task_id

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None

    def callback_metric(self, metric_name, metric_namespace, data_info):
        self.tracker.set_metric_meta(metric_namespace,
                                     metric_name,
                                     MetricMeta(name='reader',
                                                metric_type='data_info',
                                                extra_metas=data_info))

